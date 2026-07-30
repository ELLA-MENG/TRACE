from pathlib import Path
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import morphology


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "01_keypoint_detection"
INPUT = ROOT / "data/input/calibri/12pt"
OUTPUT = ROOT / "runs/calibri_12pt_watermark/01_keypoints"
CHECKPOINT_DIR = ROOT / "checkpoints/keypoint"
HEIGHT_SIZE = 368
SCORE_THRESHOLD = 0.5

sys.path.insert(0, str(STAGE))
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints
from modules.load_state import load_state
from val import normalize, pad_width


def resolve_checkpoint():
    configured = os.environ.get("TRACE_KEYPOINT_CHECKPOINT")
    if configured:
        checkpoint = Path(configured).expanduser().resolve()
        if checkpoint.is_file():
            return checkpoint
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    candidates = sorted(CHECKPOINT_DIR.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoint found in: {CHECKPOINT_DIR}")
    return candidates[0]


def nearest_black(image, point, radius=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 128)
    if len(xs) == 0:
        return None
    x, y = point
    keep = (np.abs(xs - x) <= radius) & (np.abs(ys - y) <= radius)
    if not keep.any():
        return None
    xs, ys = xs[keep], ys[keep]
    index = np.argmin((xs - x) ** 2 + (ys - y) ** 2)
    return int(xs[index]), int(ys[index])


def infer(net, image):
    stride, upsample_ratio = 8, 4
    height = image.shape[0]
    scale = HEIGHT_SIZE * 0.6 / height
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    minimum = [HEIGHT_SIZE, max(scaled.shape[1], HEIGHT_SIZE)]
    padded, pad = pad_width(scaled, stride, (256, 256, 256), minimum)
    normalized = normalize(padded, np.array([128, 128, 128], np.float32), np.float32(1 / 256))
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().cuda()
    with torch.no_grad():
        output = net(tensor)
    heatmaps = np.transpose(output[-2].squeeze().cpu().numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, None, fx=upsample_ratio * 2, fy=upsample_ratio * 2,
                          interpolation=cv2.INTER_CUBIC)
    return heatmaps, scale, pad


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint()
    net = PoseEstimationWithMobileNet()
    load_state(net, torch.load(checkpoint, map_location="cpu"))
    net = net.eval().cuda()
    summary = []
    for path in sorted(INPUT.glob("ori12_*.png")):
        original = cv2.imread(str(path))
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(255 - gray, 128, 1, cv2.THRESH_BINARY)[1]
        skeleton = 1 - morphology.skeletonize(binary).astype(np.uint8)
        h, w = original.shape[:2]
        if w < h:
            working = cv2.resize(skeleton * 255, (512, round(512 * h / w)))
            display = cv2.resize(original, (512, round(512 * h / w)))
        else:
            working = cv2.resize(skeleton * 255, (round(512 * w / h), 512))
            display = cv2.resize(original, (round(512 * w / h), 512))
        working = cv2.cvtColor(cv2.threshold(working, 128, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_GRAY2BGR)
        heatmaps, scale, pad = infer(net, working)
        edge_points, cross_points = [], []
        overlay = display.copy()
        all_by_type, total = [], 0
        for kind in range(2):
            count, all_by_type, raw = extract_keypoints(heatmaps[:, :, kind], all_by_type, total)
            total += count
            for x, y, score, _ in raw:
                if score < SCORE_THRESHOLD:
                    continue
                mapped = (round((x - pad[1]) / scale), round((y - pad[0]) / scale))
                point = nearest_black(display, mapped)
                if point is None:
                    continue
                destination = edge_points if kind == 0 else cross_points
                if point not in destination:
                    destination.append(point)
                    color = (0, 180, 0) if kind == 0 else (0, 0, 255)
                    cv2.circle(overlay, point, 7, color, -1)
        item_dir = OUTPUT / path.stem
        item_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(item_dir / "skeleton.png"), working)
        cv2.imwrite(str(item_dir / "keypoints.png"), overlay)
        heat = np.maximum(heatmaps[:, :, 0], heatmaps[:, :, 1])
        heat = np.uint8(np.clip(heat, 0, 1) * 255)
        cv2.imwrite(str(item_dir / "heatmap.png"), cv2.applyColorMap(heat, cv2.COLORMAP_JET))
        record = {"image": str(path), "working_size": [working.shape[1], working.shape[0]],
                  "edge_points": edge_points, "cross_points": cross_points,
                  "checkpoint": str(checkpoint), "threshold": SCORE_THRESHOLD}
        (item_dir / "keypoints.json").write_text(json.dumps(record, indent=2))
        summary.append({"name": path.stem, "edges": len(edge_points), "crosses": len(cross_points)})
        print(path.stem, "edges", len(edge_points), "crosses", len(cross_points), flush=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
