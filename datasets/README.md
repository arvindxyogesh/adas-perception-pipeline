# Dataset Guide

Use one of these ADAS datasets:

1. KITTI: https://www.cvlibs.net/datasets/kitti/
2. BDD100K: https://bdd-data.berkeley.edu/
3. nuScenes: https://www.nuscenes.org/

## Recommended quickstart for this pipeline
- Download a short front-camera driving clip from BDD100K or KITTI raw data.
- Save it as `datasets/sample_drive.mp4`.
- Keep clip length around 1 to 3 minutes for fast local testing.

## Automated downloader

Run from repo root:

```bash
bash scripts/download_datasets.sh
```

What it does:
- Downloads sample driving video to `datasets/sample_drive.mp4`.
- Downloads KITTI devkit (`devkit_object.zip`).
- Downloads and verifies nuScenes mini archive.
- Attempts BDD100K labels from official URL.
- If BDD100K official URL fails, tries Kaggle CLI fallback.

Kaggle fallback notes:
- Install Kaggle CLI and configure credentials at `~/.kaggle/kaggle.json`.
- Optional dataset ID list override (comma-separated):

```bash
KAGGLE_BDD100K_DATASETS=<owner1/dataset1>,<owner2/dataset2> bash scripts/download_datasets.sh
```

Download status is logged to:
- `datasets/external/DOWNLOAD_STATUS.txt`

## Optional annotation export
For precision and recall evaluation, export frame-level labels to:
`datasets/ground_truth.jsonl`

Expected format (one JSON object per line):

{"frame_id":"<uuid>","objects":[{"label":"car","bbox":[x1,y1,x2,y2]}]}
