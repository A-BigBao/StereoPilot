import json
import toml
import os
import argparse
import torch

from utils.common import DTYPE_MAP, cache_video


def read_prompt(filename):
    """Read prompt from file"""
    with open(filename, 'r', encoding='utf-8') as file:
        return file.readline().strip()


def set_config_defaults(config):
    """Set config defaults"""
    model_config = config['model']
    model_config['dtype'] = DTYPE_MAP[model_config['dtype']]
    if 'transformer_dtype' in model_config:
        model_config['transformer_dtype'] = DTYPE_MAP[model_config['transformer_dtype']]


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Stereo video generation inference")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--input", type=str, required=True, help="Input video path (mp4)")
    parser.add_argument("--output_folder", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}")

    # Derive paths
    input_path = args.input
    prompt_path = input_path.rsplit('.mp4', 1)[0] + '.txt'
    output_path = os.path.join(args.output_folder, os.path.basename(input_path))

    # Check if files exist
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    # Load config
    with open(args.config) as f:
        config = json.loads(json.dumps(toml.load(f)))
    set_config_defaults(config)
    
    # Load model
    print("Loading model...")
    from models import StereoPilot
    model = StereoPilot.StereoPilotPipeline(config)
    model.load_diffusion_model()
    model.register_custom_op()

    # Set to inference mode
    model.transformer.eval()
    torch.set_grad_enabled(False)
    
    # Move model to device
    model.transformer.to(device)
    model.vae.model.to(device)
    model.vae.mean = model.vae.mean.to(device)
    model.vae.std = model.vae.std.to(device)
    model.text_encoder.model.to(device)

    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # Read prompt
    prompt = read_prompt(prompt_path)
    print(f"Input: {input_path}")
    print(f"Prompt: {prompt_path}")
    print(f"Output: {output_path}")
    
    # Run inference
    video = model.sample(
        prompt=prompt,
        video_condition=input_path,
        size=(832, 480),
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=30,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        domain_label=1
    )

    # Save video
    cache_video(
        tensor=video[None],
        save_file=output_path,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )

    print("Inference completed!")
