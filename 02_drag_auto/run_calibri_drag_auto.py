from pathlib import Path
import copy
import json
import sys

import cv2
import numpy as np
from skimage import morphology


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/input/calibri/12pt"
KEYPOINTS = ROOT / "runs/calibri_12pt_watermark/01_keypoints"
OUTPUT = ROOT / "runs/calibri_12pt_watermark/02_drag_auto_shift20"
STAGE3 = ROOT / "03_drag_diffusion"
BIT = "1"
SHIFT_PIXELS = 20

sys.path.insert(0, str(STAGE3))
from utils.ui_utils2 import (choose_handle_points, choose_mask, choose_target_points,
                             relocate_edgepoints, relocate_refpoints)


def working_image(path):
    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    if w < h:
        return cv2.resize(image, (512, round(512 * h / w)))
    return cv2.resize(image, (round(512 * w / h), 512))


def select_character(image, edge_points, cross_points):
    if not edge_points:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    skeleton = 1 - morphology.skeletonize(1 - binary // 255).astype(np.uint8)
    skeleton_binary = skeleton * 255
    corrected = []
    valid_edges = []
    for point in edge_points:
        relocated = relocate_edgepoints(tuple(point), binary, skeleton_binary)
        if relocated is not None:
            valid_edges.append(tuple(point))
            corrected.append(tuple(map(int, relocated)))
    if not valid_edges:
        return None
    candidates = choose_handle_points(binary, valid_edges, [tuple(x) for x in cross_points], 90)
    if not candidates:
        return None
    maximum = max(item[2] for item in candidates)
    if maximum < 3:
        return None
    best = sorted((x for x in candidates if x[2] == maximum), key=lambda x: (x[0][1], x[0][0]))[0]
    original_handle, reference = tuple(best[0]), tuple(best[1])
    index = valid_edges.index(original_handle)
    handle = corrected[index]
    target_axis = int(np.argmin(np.abs(np.array(handle) - np.array(reference))))
    reference = relocate_refpoints(binary, skeleton_binary, original_handle, reference, target_axis)
    if reference is None:
        return None
    return handle, tuple(map(int, reference)), target_axis, maximum, skeleton_binary


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "data/direction").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/mask").mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted(INPUT.glob("ori12_*.png")):
        item = path.stem
        keypoint_file = KEYPOINTS / item / "keypoints.json"
        record = json.loads(keypoint_file.read_text())
        image = working_image(path)
        selected = select_character(image, record["edge_points"], record["cross_points"])
        item_dir = OUTPUT / item
        item_dir.mkdir(exist_ok=True)
        if selected is None:
            result = {"name": item, "status": "not_suitable", "reason": "automatic selection found no valid pair"}
            (item_dir / "selection.json").write_text(json.dumps(result, indent=2))
            summary.append(result)
            print(item, "not_suitable", flush=True)
            continue
        handle, reference, target_axis, score, skeleton = selected
        direction_canvas = image.copy()
        try:
            target, handle, slope = choose_target_points(direction_canvas, handle, reference, item,
                                                          target_axis, SHIFT_PIXELS, BIT)
            target_method = "original_choose_target_points"
        except (ValueError, IndexError, ZeroDivisionError):
            # Some thin glyphs do not intersect the legacy fixed-radius probe.
            # Preserve its intended radial move using the handle-reference vector.
            vector = np.array(handle, dtype=float) - np.array(reference, dtype=float)
            length = np.linalg.norm(vector)
            if length == 0:
                result = {"name": item, "status": "not_suitable", "reason": "zero direction vector"}
                (item_dir / "selection.json").write_text(json.dumps(result, indent=2))
                summary.append(result)
                continue
            unit = vector / length
            target_array = np.array(handle, dtype=float) + unit * SHIFT_PIXELS
            target = (int(np.clip(round(target_array[0]), 1, image.shape[1] - 1)),
                      int(np.clip(round(target_array[1]), 1, image.shape[0] - 1)))
            slope = tuple(map(int, vector))
            target_method = "radial_fallback"
        binary3 = cv2.cvtColor(cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 128, 255,
                                             cv2.THRESH_BINARY)[1], cv2.COLOR_GRAY2BGR)
        mask, _, mask_list = choose_mask(binary3, handle, target, target_axis, item, slope)
        mask_u8 = np.uint8(mask > 0) * 255
        overlay = image.copy()
        red = np.zeros_like(overlay); red[:, :, 2] = 255
        overlay = np.where((mask_u8[:, :, None] > 0), cv2.addWeighted(overlay, .55, red, .45, 0), overlay)
        points = image.copy()
        cv2.circle(points, tuple(handle), 8, (0, 180, 0), -1)
        cv2.circle(points, tuple(reference), 8, (0, 200, 255), -1)
        cv2.circle(points, tuple(target), 8, (0, 0, 255), -1)
        cv2.arrowedLine(points, tuple(handle), tuple(target), (255, 0, 0), 4, tipLength=.25)
        cv2.imwrite(str(item_dir / "skeleton.png"), skeleton)
        cv2.imwrite(str(item_dir / "selected_points.png"), points)
        cv2.imwrite(str(item_dir / "mask.png"), mask_u8)
        cv2.imwrite(str(item_dir / "mask_overlay.png"), overlay)
        result = {"name": item, "status": "selected", "bit": BIT, "handle_point": handle,
                  "reference_point": reference, "target_point": target, "target_axis": target_axis,
                  "selection_score": float(score), "slope": list(map(int, slope)),
                  "target_method": target_method, "shift_pixels": SHIFT_PIXELS,
                  "mask_box": list(map(int, mask_list))}
        (item_dir / "selection.json").write_text(json.dumps(result, indent=2))
        summary.append(result)
        print(item, "selected", handle, "->", target, flush=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("selected", sum(x["status"] == "selected" for x in summary), "of", len(summary))


if __name__ == "__main__":
    main()
