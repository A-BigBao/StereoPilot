import torch
import torchvision
import imageio


DTYPE_MAP = {
    'float32': torch.float32, 
    'float16': torch.float16, 
    'bfloat16': torch.bfloat16, 
    'float8': torch.float8_e4m3fn
}

VIDEO_EXTENSIONS = set(x.extension for x in imageio.config.video_extensions)

AUTOCAST_DTYPE = None


def cache_video(tensor, save_file=None, fps=30, nrow=8, normalize=True, value_range=(-1, 1), retry=5):
    """
    Save video tensor to file
    
    Args:
        tensor: Video tensor [B, C, T, H, W]
        save_file: Save path
        fps: Frame rate
        nrow: Number of rows in grid
        normalize: Whether to normalize
        value_range: Value range
        retry: Number of retries
    """
    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack([
                torchvision.utils.make_grid(u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ], dim=1).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            writer = imageio.get_writer(save_file, fps=fps, codec='libx264', quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return save_file
        except Exception as e:
            error = e
            continue
    
    print(f'cache_video failed, error: {error}', flush=True)
    return None


def round_to_nearest_multiple(x, multiple):
    """Round to nearest multiple"""
    return int(round(x / multiple) * multiple)


def round_down_to_multiple(x, multiple):
    """Round down to multiple"""
    return int((x // multiple) * multiple)
