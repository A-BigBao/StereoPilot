from models.wan import *
from models.base import PreprocessMediaFile

import torch
from safetensors.torch import load_file


def register_model(model):
    """Register custom model forward pass"""
    def create_custom_model_forward(self):
        def custom_forward(x, t, context, seq_len, clip_fea=None, y=None, domain_label=0):
            """
            Forward pass through diffusion model
            
            Args:
                x: Input video tensor list
                t: Diffusion timestep
                context: Text embedding list
                seq_len: Max sequence length
                clip_fea: CLIP image features (optional)
                y: Conditional video input (optional)
                domain_label: Domain label (0=stereo4d, 1=3dmovie)
            """
            device = self.patch_embedding.weight.device
            if self.freqs.device != device:
                self.freqs = self.freqs.to(device)

            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            assert seq_lens.max() <= seq_len
            x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
            ])
        
            # Time embedding
            with amp.autocast(dtype=torch.float32):
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            
            # Domain embedding
            if domain_label == 0:
                domain_emb = self.parall_embedding.unsqueeze(0)
            else:
                domain_emb = self.converge_embedding.unsqueeze(0)
            e0 = e0 + domain_emb.to(e0.dtype)
            
            # Text context
            context = self.text_embedding(
                torch.stack([
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ])
            )

            if clip_fea is not None:
                context_clip = self.img_emb(clip_fea)
                context = torch.concat([context_clip, context], dim=1)

            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=None
            )

            for block in self.blocks:
                x = block(x, **kwargs)

            x = self.head(x, e)
            x = self.unpatchify(x, grid_sizes)
            return [u.float() for u in x]
        
        return custom_forward
    
    model.forward = create_custom_model_forward(model)


class StereoPilotPipeline(WanPipeline):
    """StereoPilot Pipeline for stereo video generation"""
    name = 'stereopilot'
    framerate = 16

    def __init__(self, config):
        super().__init__(config)

    def load_diffusion_model(self):
        """Load diffusion model and domain embeddings"""
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        if transformer_path := self.model_config.get('transformer_path', None):
            self.transformer = WanModelFromSafetensors.from_pretrained(
                transformer_path,
                self.original_model_config_path,
                torch_dtype=dtype,
                transformer_dtype=transformer_dtype,
            )
            print('Transformer loaded from', transformer_path)
        else:
            self.transformer = WanModel.from_pretrained(self.model_config['ckpt_path'], torch_dtype=dtype)
            for name, p in self.transformer.named_parameters():
                if not (any(x in name for x in KEEP_IN_HIGH_PRECISION)):
                    p.data = p.data.to(transformer_dtype)

        # Add domain embedding parameters
        self.transformer.parall_embedding = torch.nn.Parameter(torch.zeros(6, 1536, device="cuda", dtype=dtype))
        self.transformer.converge_embedding = torch.nn.Parameter(torch.zeros(6, 1536, device="cuda", dtype=dtype))

        # Load pretrained weights
        if pretrained_path := self.model_config.get('pretrained_path', None):
            print('Loading weights from', pretrained_path)
            state_dict = load_file(pretrained_path)
            self.transformer.load_state_dict(state_dict)
            for name, p in self.transformer.named_parameters():
                if not (any(x in name for x in KEEP_IN_HIGH_PRECISION)):
                    p.data = p.data.to(transformer_dtype)

    @torch.no_grad()
    def sample(self, video_condition, prompt=None, context=None, context_lens=None,
               size=(832, 480), frame_num=81, shift=5.0, sample_solver='unipc',
               sampling_steps=50, guide_scale=5.0, n_prompt="", seed=-1, domain_label=1):
        """
        Generate stereo video
        
        Args:
            video_condition: Input video path or tensor
            prompt: Text prompt
            context: Pre-computed text embeddings
            context_lens: Context lengths
            size: Output size
            frame_num: Number of frames
            shift: Scheduler shift
            sample_solver: Sampler type
            sampling_steps: Sampling steps
            guide_scale: Guidance scale
            n_prompt: Negative prompt
            seed: Random seed
            domain_label: Domain label
        """
        device = 'cuda'
        self.transformer.eval()
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

        # Process video condition
        if isinstance(video_condition, str):
            preprocessor = self.get_preprocess_media_file_fn()
            tensor = preprocessor((None, video_condition), mask_filepath=None, size_bucket=(832, 480, 81))[0][0].unsqueeze(0)
            latents_video_condition = self.vae.model.encode(tensor.to(device), self.vae.scale).squeeze()
        elif isinstance(video_condition, torch.Tensor):
            latents_video_condition = video_condition.squeeze(0).to(device)
        
        target_shape = latents_video_condition.shape
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                           (self.patch_size[1] * self.patch_size[2]) *
                           target_shape[1] / self.sp_size) * self.sp_size

        # Set random seed
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        
        # Process text context
        if prompt is not None:
            context = self.text_encoder([prompt], device)
        else:
            context = [emb[:length].to(device) for emb, length in zip(context, context_lens)]

        noise = [latents_video_condition]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.transformer, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.model_config['dtype']), torch.no_grad(), no_sync():
            latents = noise
            arg_c = {'context': context, 'seq_len': seq_len}

            latent_model_input = latents
            timestep = torch.tensor([1], device=device)
            
            noise_pred_cond = self.transformer(
                latent_model_input, t=timestep, domain_label=domain_label, **arg_c
            )[0]
            
            x0 = noise_pred_cond.unsqueeze(0)
            videos = self.vae.decode(x0)

        return videos[0]

    def register_custom_op(self):
        """Register custom operations"""
        register_model(self.transformer)
