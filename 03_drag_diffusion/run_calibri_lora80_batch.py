from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "runs/calibri_12pt_watermark/02_drag_auto_shift20"
OUTPUT = ROOT / "runs/calibri_12pt_watermark/03_drag_diffusion_shift20_latent001_lora80_lr001_lam5_inv09"
RUNNER = ROOT / "03_drag_diffusion/run_calibri_dragdiffusion.py"

selected = []
for path in sorted(SELECTION.glob("ori12_*/selection.json")):
    if json.loads(path.read_text()).get("status") == "selected":
        selected.append(path.parent.name)

OUTPUT.mkdir(parents=True, exist_ok=True)
summary = []
for index, name in enumerate(selected, 1):
    print(f"[{index}/{len(selected)}] {name}", flush=True)
    log = OUTPUT / f"{name}.log"
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    with log.open("w") as stream:
        completed = subprocess.run(
            [sys.executable, str(RUNNER), "--only", name, "--steps", "1000"],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
        )
    result_path = OUTPUT / name / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {
        "name": name, "status": "failed", "error": f"child exit code {completed.returncode}"
    }
    summary.append(result)
    (OUTPUT / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    # Child-process exit releases its complete CUDA context before the next character.
    time.sleep(2)
print("finished", len(summary), flush=True)
