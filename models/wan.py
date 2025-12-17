import sys
import json
import math
import re
import os.path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/Wan2_1'))

import torch
from torch import nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE
import wan
from wan.modules.t5 import T5Encoder
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae import WanVAE
from wan.modules.model import (
    WanModel, sinusoidal_embedding_1d, WanLayerNorm, WanSelfAttention, WAN_CROSSATTENTION_CLASSES
)
from wan import configs as wan_configs
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import random
from contextlib import contextmanager
import torch.cuda.amp as amp
from safetensors.torch import load_file

from utils.common import cache_video

KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'head', 'modulation']


class WanModelFromSafetensors(WanModel):
    """Load WanModel from safetensors file"""
    
    @classmethod
    def from_pretrained(cls, weights_file, config_file, torch_dtype=torch.bfloat16, transformer_dtype=torch.bfloat16):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)

        with init_empty_weights():
            model = cls(**config)

        state_dict = load_file(weights_file, device='cpu')
        state_dict = {
            re.sub(r'^model\.diffusion_model\.', '', k): v for k, v in state_dict.items()
        }

        for name, param in model.named_parameters():
            dtype_to_use = torch_dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
            set_module_tensor_to_device(model, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        return model


def umt5_keys_mapping_comfy(state_dict):
    """UMT5 weights mapping for ComfyUI format"""
    def execute_mapping(original_key):
        if original_key == "shared.weight":
            return "token_embedding.weight"
        if original_key == "encoder.final_layer_norm.weight":
            return "norm.weight"

        block_match = re.match(r"encoder\.block\.(\d+)\.layer\.(\d+)\.(.+)", original_key)
        if block_match:
            block_num = block_match.group(1)
            layer_type = int(block_match.group(2))
            rest = block_match.group(3)

            if layer_type == 0:
                if "SelfAttention" in rest:
                    attn_part = rest.split(".")[1]
                    if attn_part in ["q", "k", "v", "o"]:
                        return f"blocks.{block_num}.attn.{attn_part}.weight"
                    elif attn_part == "relative_attention_bias":
                        return f"blocks.{block_num}.pos_embedding.embedding.weight"
                elif rest == "layer_norm.weight":
                    return f"blocks.{block_num}.norm1.weight"
            elif layer_type == 1:
                if "DenseReluDense" in rest:
                    parts = rest.split(".")
                    if parts[1] == "wi_0":
                        return f"blocks.{block_num}.ffn.gate.0.weight"
                    elif parts[1] == "wi_1":
                        return f"blocks.{block_num}.ffn.fc1.weight"
                    elif parts[1] == "wo":
                        return f"blocks.{block_num}.ffn.fc2.weight"
                elif rest == "layer_norm.weight":
                    return f"blocks.{block_num}.norm2.weight"
        return None

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = execute_mapping(key)
        if new_key:
            new_state_dict[new_key] = value
    del state_dict
    return new_state_dict


def umt5_keys_mapping_kijai(state_dict):
    """UMT5 weights mapping for Kijai format"""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("attention.", "attn.")
        new_key = new_key.replace("final_norm.weight", "norm.weight")
        new_state_dict[new_key] = value
    del state_dict
    return new_state_dict


def umt5_keys_mapping(state_dict):
    """Auto-detect and map UMT5 weights"""
    if 'blocks.0.attn.k.weight' in state_dict:
        return umt5_keys_mapping_kijai(state_dict)
    else:
        return umt5_keys_mapping_comfy(state_dict)


def _t5(name, encoder_only=False, return_tokenizer=False, tokenizer_kwargs={}, dtype=torch.float32, device='cpu', **kwargs):
    """Build T5 model"""
    if encoder_only:
        from wan.modules.t5 import T5Encoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('encoder_layers')
        _ = kwargs.pop('decoder_layers')
        model_cls = T5Encoder

    with torch.device(device):
        model = model_cls(**kwargs)

    if return_tokenizer:
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        return model, tokenizer
    return model


def umt5_xxl(**kwargs):
    """Build UMT5-XXL model"""
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1
    )
    cfg.update(**kwargs)
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:
    """T5 text encoder"""

    def __init__(self, text_len, dtype=torch.bfloat16, device='cpu', checkpoint_path=None, tokenizer_path=None, shard_fn=None):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device

        with init_empty_weights():
            model = umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=dtype,
                device=device
            ).eval().requires_grad_(False)

        if checkpoint_path.endswith('.safetensors'):
            state_dict = load_file(checkpoint_path, device='cpu')
            state_dict = umt5_keys_mapping(state_dict)
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(state_dict, assign=True)
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts, device):
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]


class WanAttentionBlock(nn.Module):
    """Wan attention block"""

    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, window_size=(-1, -1), 
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), 
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        e = (self.modulation + e).chunk(6, dim=1)
        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)
        x = x + y * e[2]

        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x


class Head(nn.Module):
    """Output head"""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        with torch.autocast('cuda', dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


# Replace modules
wan.modules.model.WanAttentionBlock = WanAttentionBlock
wan.modules.model.Head = Head


class WanPipeline(BasePipeline):
    """Wan Pipeline base class"""
    name = 'wan'
    framerate = 16

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        ckpt_dir = self.model_config['ckpt_path']
        dtype = self.model_config['dtype']

        self.original_model_config_path = os.path.join(ckpt_dir, 'config.json')
        with open(self.original_model_config_path) as f:
            json_config = json.load(f)
        self.i2v = (json_config['model_type'] == 'i2v')
        model_dim = json_config['dim']
        
        if not self.i2v and model_dim == 1536:
            wan_config = wan_configs.t2v_1_3B
        elif self.i2v and model_dim == 5120:
            wan_config = wan_configs.i2v_14B
        elif not self.i2v and model_dim == 5120:
            wan_config = wan_configs.t2v_14B
        else:
            raise RuntimeError(f'Could not autodetect model variant. model_dim={model_dim}, i2v={self.i2v}')
        
        self.wan_config = wan_config
        t5_model_path = self.model_config.get('llm_path') or os.path.join(ckpt_dir, wan_config.t5_checkpoint)
        
        self.text_encoder = T5EncoderModel(
            text_len=wan_config.text_len,
            dtype=dtype,
            device='cpu',
            checkpoint_path=t5_model_path,
            tokenizer_path=os.path.join(ckpt_dir, wan_config.t5_tokenizer),
            shard_fn=None,
        )

        self.vae = WanVAE(
            vae_pth=os.path.join(ckpt_dir, wan_config.vae_checkpoint),
            device='cpu',
        )
        self.vae.mean = self.vae.mean.to('cuda')
        self.vae.std = self.vae.std.to('cuda')
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

        self.vae_stride = wan_config.vae_stride
        self.patch_size = wan_config.patch_size
        self.sp_size = 1

    def load_diffusion_model(self):
        """Load diffusion model"""
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        if transformer_path := self.model_config.get('transformer_path', None):
            self.transformer = WanModelFromSafetensors.from_pretrained(
                transformer_path,
                self.original_model_config_path,
                torch_dtype=dtype,
                transformer_dtype=transformer_dtype,
            )
        else:
            self.transformer = WanModel.from_pretrained(self.model_config['ckpt_path'], torch_dtype=dtype)
            for name, p in self.transformer.named_parameters():
                if not (any(x in name for x in KEEP_IN_HIGH_PRECISION)):
                    p.data = p.data.to(transformer_dtype)

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
            round_height=8,
            round_width=8,
            round_frames=4,
        )
