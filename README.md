# TRACE

Official implementation of **TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking**.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Keypoint Network Training](#keypoint-network-training)
- [Pipeline](#pipeline)
- [Configuration](#configuration)
- [Acknowledgements](#acknowledgements)

---

## Installation

TRACE requires Linux and an NVIDIA GPU with approximately 14 GB of memory.

```bash
conda env create -f environment.yaml
conda activate trace
```

Pretrained model files are not included in this repository. Place a pretrained
keypoint detector (`.pth`) in `checkpoints/keypoint/` before running Stage 1.
Stable Diffusion v1.5 is downloaded automatically on first use.

---

## Quick Start

Several 12 pt test character images are included in:

```text
data/input/calibri/12pt/
```

Run the complete pipeline with one command:

```bash
python run_trace.py
```

Results are saved under:

```text
runs/calibri_12pt_watermark/
```

To run one stage only:

```bash
python run_trace.py --stage 1  # keypoint detection
python run_trace.py --stage 2  # automatic point selection
python run_trace.py --stage 3  # diffusion editing
```

---

## Keypoint Network Training

Organize your training data as:

```text
data/keypoint/
└── train/
    ├── images/
    │   ├── sample_0001.png
    │   └── sample_0002.png
    └── prepared_annotations.pkl
```

`prepared_annotations.pkl` is a Python list of dictionaries. Each dictionary
describes one character image and must contain:

```python
{
    "img_paths": "sample_0001.png",
    "img_width": 128,
    "img_height": 128,
    "objpos": [64, 64],
    "bbox": [y_min, y_max, x_min, x_max],
    "scale_provided": 0.35,
    "num_keypoints": 6,
    "keypoints": [
        [[x, y, visibility], ...],  # endpoints
        [[x, y, visibility], ...]   # junctions
    ],
    "processed_other_annotations": []
}
```

Coordinates use the original image resolution. `visibility=2` denotes a valid
visible keypoint. `img_paths` is resolved relative to `train/images/`.

Start training from scratch:

```bash
python 01_keypoint_detection/train_keypoint.py \
    --train-images-folder data/keypoint/train/images \
    --prepared-train-labels data/keypoint/train/prepared_annotations.pkl \
    --experiment-name trace_keypoint \
    --batch-size 32 \
    --base-lr 9e-4 \
    --checkpoint-after 1000
```

Saved checkpoints are written to:

```text
trace_keypoint_checkpoints/
```

After training, place the selected `.pth` file in:

```text
checkpoints/keypoint/
```

---

## Pipeline

TRACE contains three stages:

1. **Keypoint Detection** - detects skeleton endpoints and junctions.
2. **Drag Auto** - selects the handle point, target point, and editable mask.
3. **Drag Diffusion** - trains a per-image LoRA and performs point-guided editing.

```text
input image
   -> keypoints
   -> handle / target / mask
   -> watermarked image
```

The final output uses the edited image inside the mask and preserves the original image outside the mask. Otsu thresholding is used to obtain the binary result.

---

## Configuration

The complete configuration is available in `configs/pipeline.yaml`.

---

## Acknowledgements

The diffusion editing stage is based on [DragDiffusion](https://github.com/Yujun-Shi/DragDiffusion). Its upstream README and Apache-2.0 license are retained in `03_drag_diffusion/`.
