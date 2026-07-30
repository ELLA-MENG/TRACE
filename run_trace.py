#!/usr/bin/env python3
"""Run the three TRACE stages in sequence."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data/input/calibri/12pt"
CHECKPOINT_DIR = ROOT / "checkpoints/keypoint"

STAGES = {
    "1": ROOT / "01_keypoint_detection/run_calibri_keypoints.py",
    "2": ROOT / "02_drag_auto/run_calibri_drag_auto.py",
    "3": ROOT / "03_drag_diffusion/run_calibri_lora80_batch.py",
}


def find_checkpoint(checkpoint=None):
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    candidates = sorted(CHECKPOINT_DIR.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(
            "Pretrained weights are not included. Place a .pth file in "
            f"{CHECKPOINT_DIR} or pass --checkpoint /path/to/model.pth"
        )
    return candidates[0]


def preflight(needs_stage_1, checkpoint=None):
    images = sorted(INPUT_DIR.glob("ori12_*.png"))
    if not images:
        raise FileNotFoundError(f"No input images found in: {INPUT_DIR}")
    print(f"Found {len(images)} input image(s).")
    return find_checkpoint(checkpoint) if needs_stage_1 else None


def run(stage, checkpoint=None):
    script = STAGES[stage]
    print(f"\n[TRACE] Stage {stage}/3: {script.parent.name}", flush=True)
    env = os.environ.copy()
    if checkpoint:
        env["TRACE_KEYPOINT_CHECKPOINT"] = str(checkpoint)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the TRACE watermarking pipeline.")
    parser.add_argument(
        "--stage",
        choices=["all", "1", "2", "3"],
        default="all",
        help="Run all stages or a single stage (default: all).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to a pretrained keypoint detector (.pth).",
    )
    args = parser.parse_args()
    selected = ["1", "2", "3"] if args.stage == "all" else [args.stage]
    checkpoint = preflight("1" in selected, args.checkpoint)
    for stage in selected:
        run(stage, checkpoint)
    print("\n[TRACE] Finished. Results are under runs/calibri_12pt_watermark/", flush=True)


if __name__ == "__main__":
    main()
