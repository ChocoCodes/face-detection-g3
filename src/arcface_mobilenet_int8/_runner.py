import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_base(script_name: str, passthrough_args: list[str]) -> int:
    base_script = PROJECT_ROOT / "src" / "arcface_mobilenet" / script_name
    model_dir = PROJECT_ROOT / "models" / "arcface_mobilenet_int8"
    enrollment_path = model_dir / "enrollment.json"
    enrollment_flag = "--enrollment-output" if script_name == "trainer.py" else "--enrollment-path"

    cmd = [
        sys.executable,
        str(base_script),
        "--model-dir",
        str(model_dir),
        enrollment_flag,
        str(enrollment_path),
        *passthrough_args,
    ]
    return subprocess.call(cmd)
