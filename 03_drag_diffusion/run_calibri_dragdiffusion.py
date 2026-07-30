from pathlib import Path
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "03_drag_diffusion"
INPUT = ROOT / "data/input/calibri/12pt"
SELECTION = ROOT / "runs/calibri_12pt_watermark/02_drag_auto_shift20"
OUTPUT = ROOT / "runs/calibri_12pt_watermark/03_drag_diffusion_shift20_latent001_lora80_lr001_lam5_inv09"
MODEL = "runwayml/stable-diffusion-v1-5"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(STAGE))
from utils.ui_utils2 import run_drag
from utils.lora_utils import train_lora


def working_image(path):
    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    if w < h:
        return cv2.resize(image, (512, round(512 * h / w)))
    return cv2.resize(image, (round(512 * w / h), 512))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected characters")
    parser.add_argument("--steps", type=int, default=1000, help="Outer TRACE n_pix_step")
    parser.add_argument("--only", type=str, default="", help="Run one stem, e.g. ori12_00")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = []
    for path in sorted(INPUT.glob("ori12_*.png")):
        record_path = SELECTION / path.stem / "selection.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text())
        if record.get("status") == "selected":
            selected.append((path, record))
    if args.only:
        selected = [x for x in selected if x[0].stem == args.only]
    elif args.limit:
        selected = selected[:args.limit]

    summary = []
    for index, (path, selection) in enumerate(selected, 1):
        item_dir = OUTPUT / path.stem
        item_dir.mkdir(exist_ok=True)
        image = working_image(path)
        mask = cv2.imread(str(SELECTION / path.stem / "mask.png"), cv2.IMREAD_GRAYSCALE)
        handle = selection["handle_point"]
        target = selection["target_point"]
        reference = selection["reference_point"]
        started = time.time()
        print(f"[{index}/{len(selected)}] {path.stem} start", flush=True)
        try:
            lora_path = item_dir / "lora"
            lora_path.mkdir(exist_ok=True)
            train_lora(image, "binary image", MODEL, "default", str(lora_path),
                       80, 0.001, 2, 16)
            output, output_gray = run_drag(
                source_image=image,
                mask=mask,
                prompt="binary image",
                points=[handle, target],
                ref_points=reference,
                inversion_strength=0.9,
                lam=5,
                latent_lr=0.01,
                n_pix_step=args.steps,
                model_path=MODEL,
                vae_path="default",
                lora_path=str(lora_path),
                start_step=0,
                start_layer=10,
                save_dir=str(item_dir),
                masklist=selection["mask_box"],
            )
            generated_binary = cv2.threshold(output_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            original_binary = cv2.threshold(original_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            mask_boolean = mask > 0
            # Correct legacy behavior: generated pixels only inside mask; preserve original elsewhere.
            composite_gray = np.where(mask_boolean, output_gray, original_gray).astype(np.uint8)
            binary = np.where(mask_boolean, generated_binary, original_binary).astype(np.uint8)
            original = cv2.imread(str(path))
            final = cv2.resize(binary, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(item_dir / "dragdiffusion_rgb.png"), output)
            cv2.imwrite(str(item_dir / "watermarked_composite_gray.png"), composite_gray)
            cv2.imwrite(str(item_dir / "watermarked_binary.png"), binary)
            cv2.imwrite(str(item_dir / "watermarked_original_size.png"), final)
            points_image = cv2.imread(str(SELECTION / path.stem / "selected_points.png"))
            comparison = np.hstack([image, points_image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                                    cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)])
            cv2.imwrite(str(item_dir / "process_comparison.png"), comparison)
            outside_changed = int(np.count_nonzero(composite_gray[~mask_boolean] != original_gray[~mask_boolean]))
            result = {"name": path.stem, "status": "completed", "runtime_seconds": time.time() - started,
                      "model": MODEL, "train_lora": True, "lora_step": 80, "lora_lr": 0.001,
                      "lora_batch_size": 2, "lora_rank": 16, "n_pix_step_outer": args.steps,
                      "n_pix_step_effective": 100, "inversion_strength": 0.9,
                      "latent_lr": 0.01, "lam": 5, "shift_pixels": 20,
                      "postprocess": "generated inside mask, original outside; Otsu binary",
                      "outside_mask_changed_pixels": outside_changed}
        except Exception as error:
            result = {"name": path.stem, "status": "failed", "runtime_seconds": time.time() - started,
                      "error": repr(error), "model": MODEL, "train_lora": True,
                      "lora_step": 80, "lora_lr": 0.001, "lora_batch_size": 2,
                      "lora_rank": 16, "n_pix_step_outer": args.steps,
                      "n_pix_step_effective": 100, "inversion_strength": 0.9,
                      "latent_lr": 0.01, "lam": 5, "shift_pixels": 20}
        (item_dir / "result.json").write_text(json.dumps(result, indent=2))
        summary.append(result)
        print(path.stem, result["status"], flush=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
