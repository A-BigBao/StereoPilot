# Stereo4D Dataset Processing Guide

## 1. Download Stereo4D Dataset
**Note: You need to prepare at least 9-10TB of storage**

### (1) NPZ Files Download
Please refer to: https://github.com/Stereo4d/stereo4d-code

### (2) Raw Video Download
The raw videos (StereoPliot downloads in webm format) can be downloaded using:
- https://github.com/yt-dlp/yt-dlp
- https://github.com/jinotter3/stereo4d_downloader

Thanks to these authors for their contributions. If you have any questions downloading these data, feel free to ask.

## 2. Rectify Data
**Recommendation: Use multiple machines with multi-core CPUs. Suggested configuration: 16/32 cores per machine. More machines = faster processing**

Execute the following command:
- Replace `--npz_folder` with the NPZ path downloaded in step 1
- Replace `--raw_video_folder` with the raw video path downloaded in step 1
- Replace `--output_folder` with your desired output directory
- Replace `--output_hfov` with your desired horizontal fov
- If you need other resolutions, you can modify the `NewCropFlags` class in `rectify_832.py`

```bash
python batch_rectify_832.py \
  --npz_folder "path/to/NPZ/folder" \
  --raw_video_folder "path/to/raw/video/folder" \
  --output_folder "path/to/output/folder" \
  --output_hfov "output horizontal fov" \
  --total_shards "total number of machines for processing" \
  --current_shard "current machine's shard number"
```

## 3. Captioning
For generating captions for the monocular videos, please follow the instructions provided in the **3DMovie** folder.