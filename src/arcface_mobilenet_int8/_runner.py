import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_base(script_name: str, passthrough_args: list[str]) -> int:
    base_script_name = "detect.py" if script_name == "face_detect.py" else script_name
    base_script = PROJECT_ROOT / "src" / "arcface" / base_script_name
    model_dir = PROJECT_ROOT / "models" / "arcface_mobilenet_int8"
    enrollment_path = model_dir / "enrollment.json"
    enrollment_flag = "--enrollment-output" if script_name == "trainer.py" else "--enrollment-path"

    if not model_dir.exists():
        print(f"[ERROR] Model directory not found: {model_dir}")
        print("[HINT] Build INT8 model pack first:")
        print("       python src/arcface/setup_model.py")
        print("       python src/arcface_mobilenet_int8/quantize_model.py")
        return 1

    def has_flag(flag: str) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in passthrough_args)

    extra_defaults: list[str] = []
    if script_name == "evaluate.py" and not has_flag("--report-json"):
        extra_defaults.extend(
            ["--report-json", str(PROJECT_ROOT / "reports" / "evaluation" / "arcface_mobilenet_int8_eval.json")]
        )
    if script_name == "face_detect.py":
        if not has_flag("--algorithm-name"):
            extra_defaults.extend(["--algorithm-name", "arcface_int8"])
        if not has_flag("--fps-run-prefix"):
            extra_defaults.extend(["--fps-run-prefix", "arcface_int8"])
        if not has_flag("--fps-log-path"):
            extra_defaults.extend(
                ["--fps-log-path", str(PROJECT_ROOT / "reports" / "benchmark" / "live_fps" / "arcface_int8_fps.jsonl")]
            )
        if not has_flag("--fps-summary-dir"):
            extra_defaults.extend(
                ["--fps-summary-dir", str(PROJECT_ROOT / "reports" / "benchmark" / "live_fps" / "runs")]
            )

    cmd = [
        sys.executable,
        str(base_script),
        "--model-dir",
        str(model_dir),
        enrollment_flag,
        str(enrollment_path),
        *extra_defaults,
        *passthrough_args,
    ]
    return subprocess.call(cmd)
