import shlex
import subprocess
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

GROUPED_CHOICES = [
    (
        "ArcFace",
        [
            ("setup model", "src/arcface/setup_model.py"),
            ("train enrollment", "src/arcface/trainer.py"),
            ("evaluate", "src/arcface/evaluate.py"),
            ("live detect", "src/arcface/detect.py"),
        ],
    ),
    (
        "ArcFace MobileNet INT8",
        [
            ("train enrollment", "src/arcface_mobilenet_int8/trainer.py"),
            ("evaluate", "src/arcface_mobilenet_int8/evaluate.py"),
            ("live detect", "src/arcface_mobilenet_int8/face_detect.py"),
            ("quantize model", "src/arcface_mobilenet_int8/quantize_model.py"),
        ],
    ),
    (
        "MobileFaceNet",
        [
            ("train enrollment", "src/mobilefacenet/trainer.py"),
            ("evaluate", "src/mobilefacenet/evaluate.py"),
            ("live detect", "src/mobilefacenet/detect.py"),
        ],
    ),
    (
        "EdgeFace",
        [
            ("train enrollment", "src/edgeface/trainer.py"),
            ("evaluate", "src/edgeface/evaluate.py"),
            ("live detect", "src/edgeface/face_detect.py"),
        ],
    ),
    (
        "LBPH",
        [
            ("train", "src/lbph/trainer.py"),
            ("evaluate", "src/lbph/evaluate.py"),
            ("live detect", "src/lbph/detect.py"),
        ],
    ),
    (
        "Benchmark",
        [
            ("compare models", "src/benchmark/compare_models.py"),
            ("aggregate live FPS", "src/benchmark/aggregate_live_fps.py"),
        ],
    ),
]

MODEL_INFO_CONFIG = {
    "ArcFace": {
        "trained_markers": ["models/arcface_mobilenet/enrollment.json"],
        "evaluated_reports": ["reports/evaluation/arcface_eval.json"],
        "size_paths": ["models/arcface_mobilenet"],
    },
    "ArcFace MobileNet INT8": {
        "trained_markers": ["models/arcface_mobilenet_int8/enrollment.json"],
        "evaluated_reports": ["reports/evaluation/arcface_mobilenet_int8_eval.json"],
        "size_paths": ["models/arcface_mobilenet_int8"],
    },
    "MobileFaceNet": {
        "trained_markers": ["models/yunet_mobilefacenet/enrollment.json"],
        "evaluated_reports": ["reports/evaluation/yunet_mobilefacenet_eval.json"],
        "size_paths": [
            "models/yunet_mobilefacenet/mobilefacenet.onnx",
            "models/yunet_mobilefacenet/face_detection_yunet_2023mar.onnx",
            "models/yunet_mobilefacenet/enrollment.json",
        ],
    },
    "EdgeFace": {
        "trained_markers": ["models/edgeface/enrollment.json"],
        "evaluated_reports": ["reports/evaluation/edgeface_eval.json"],
        "size_paths": [
            "models/edgeface/edgeface_xs.onnx",
            "models/edgeface/enrollment.json",
            "models/yunet_mobilefacenet/face_detection_yunet_2023mar.onnx",
        ],
    },
    "LBPH": {
        "trained_markers": [
            "models/lbph/trainer_lasalle.yml",
            "models/lbph/labels_lasalle.json",
        ],
        "evaluated_reports": ["reports/evaluation/lbph_eval.json"],
        "size_paths": [
            "models/lbph/trainer_lasalle.yml",
            "models/lbph/labels_lasalle.json",
        ],
    },
    "Benchmark": {
        "trained_markers": [],
        "evaluated_reports": [],
        "size_paths": [],
    },
}

BENCHMARK_OVERVIEW_CONFIG = {
    "ArcFace": {
        "eval_report": "reports/evaluation/arcface_eval.json",
        "fps_algorithm": "arcface",
    },
    "ArcFace MobileNet INT8": {
        "eval_report": "reports/evaluation/arcface_mobilenet_int8_eval.json",
        "fps_algorithm": "arcface_int8",
    },
    "MobileFaceNet": {
        "eval_report": "reports/evaluation/yunet_mobilefacenet_eval.json",
        "fps_algorithm": "mobilefacenet",
    },
    "EdgeFace": {
        "eval_report": "reports/evaluation/edgeface_eval.json",
        "fps_algorithm": "edgeface",
    },
    "LBPH": {
        "eval_report": "reports/evaluation/lbph_eval.json",
        "fps_algorithm": "lbph",
    },
}

def resolve_path(rel_path: str) -> Path:
    return PROJECT_ROOT / rel_path


INT8_MODEL_DIR = resolve_path("models/arcface_mobilenet_int8")
INT8_REQUIRED_MODELS = [
    resolve_path("models/arcface_mobilenet_int8/w600k_mbf.onnx"),
    resolve_path("models/arcface_mobilenet_int8/models/buffalo_s/w600k_mbf.onnx"),
]


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_hit_rate_percent(report_payload: dict | None) -> float | None:
    if not report_payload:
        return None
    threshold_sweep = report_payload.get("threshold_sweep", [])
    if isinstance(threshold_sweep, list) and threshold_sweep:
        best = None
        for row in threshold_sweep:
            try:
                value = float(row.get("overall_hit_rate_percent", -1.0))
            except Exception:
                continue
            if best is None or value > best:
                best = value
        if best is not None and best >= 0:
            return best
    overall = report_payload.get("overall", {})
    if "hit_rate_percent" in overall:
        try:
            return float(overall["hit_rate_percent"])
        except Exception:
            return None
    return None


def collect_fps_summary() -> dict[str, float]:
    aggregate_path = resolve_path("reports/benchmark/live_fps/aggregate_summary.json")
    aggregate_payload = load_json_if_exists(aggregate_path)
    if aggregate_payload:
        out: dict[str, float] = {}
        for row in aggregate_payload.get("algorithms", []):
            algo = str(row.get("algorithm", "")).strip().lower()
            if not algo:
                continue
            try:
                out[algo] = float(row.get("average_fps", 0.0))
            except Exception:
                continue
        if out:
            return out

    runs_dir = resolve_path("reports/benchmark/live_fps/runs")
    if not runs_dir.exists():
        return {}

    grouped: dict[str, dict[str, float]] = {}
    for run_file in runs_dir.glob("*.json"):
        payload = load_json_if_exists(run_file)
        if not payload:
            continue
        algo = str(payload.get("algorithm", "")).strip().lower()
        if not algo:
            continue
        frames = float(payload.get("frames", 0.0))
        duration = float(payload.get("duration_seconds", 0.0))
        avg_fps = float(payload.get("average_fps", 0.0))
        group = grouped.setdefault(algo, {"frames": 0.0, "duration": 0.0, "sum_fps": 0.0, "runs": 0.0})
        group["frames"] += max(0.0, frames)
        group["duration"] += max(0.0, duration)
        group["sum_fps"] += max(0.0, avg_fps)
        group["runs"] += 1.0

    out: dict[str, float] = {}
    for algo, row in grouped.items():
        if row["duration"] > 0 and row["frames"] > 0:
            out[algo] = row["frames"] / row["duration"]
        elif row["runs"] > 0:
            out[algo] = row["sum_fps"] / row["runs"]
    return out


def print_benchmark_overview() -> None:
    fps_by_algo = collect_fps_summary()

    print("\nOverview (all models)")
    print(f"{'Model':<28} {'Hit Rate':>10} {'Avg FPS':>10} {'Size':>12}")
    print(f"{'-'*28} {'-'*10} {'-'*10} {'-'*12}")
    for model_name, cfg in BENCHMARK_OVERVIEW_CONFIG.items():
        report_path = resolve_path(cfg["eval_report"])
        hit_rate = extract_hit_rate_percent(load_json_if_exists(report_path))
        hit_rate_display = f"{hit_rate:.2f}%" if hit_rate is not None else "N/A"

        algo_key = str(cfg["fps_algorithm"]).strip().lower()
        fps_value = fps_by_algo.get(algo_key)
        fps_display = f"{fps_value:.2f}" if fps_value is not None else "N/A"
        size_display = get_model_info(model_name)["size"]

        print(f"{model_name:<28} {hit_rate_display:>10} {fps_display:>10} {size_display:>12}")


def get_model_info(model_name: str) -> dict:
    cfg = MODEL_INFO_CONFIG.get(model_name, {})
    trained_markers = [resolve_path(p) for p in cfg.get("trained_markers", [])]
    evaluated_reports = [resolve_path(p) for p in cfg.get("evaluated_reports", [])]
    size_paths = [resolve_path(p) for p in cfg.get("size_paths", [])]

    trained = bool(trained_markers) and all(p.exists() for p in trained_markers)
    evaluated = bool(evaluated_reports) and any(p.exists() for p in evaluated_reports)
    size_bytes = sum(path_size_bytes(p) for p in size_paths)

    return {
        "trained": trained,
        "evaluated": evaluated,
        "size": format_size(size_bytes),
    }


def int8_model_pack_ready() -> bool:
    if not INT8_MODEL_DIR.exists():
        return False
    return any(p.exists() for p in INT8_REQUIRED_MODELS)


def get_python_command() -> list[str]:
    # Always prefer global interpreter so menu actions are not tied to .venv.
    configured = os.environ.get("FACE_G3_PYTHON", "").strip()
    if configured:
        return shlex.split(configured)
    return ["python"]


def run_python_script(rel_script: str, extra_args: list[str], label: str) -> int:
    script_path = resolve_path(rel_script)
    cmd = [*get_python_command(), str(script_path), *extra_args]
    print(f"\n[RUN] {label}")
    print(f"[CMD] {' '.join(shlex.quote(part) for part in cmd)}\n")
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    print(f"\n[EXIT] code={completed.returncode}")
    return completed.returncode


def maybe_prepare_int8_pack() -> bool:
    if int8_model_pack_ready():
        return True

    print("\n[INFO] ArcFace INT8 model pack is missing.")
    print("Required first-time setup:")
    print("  1) Download FP32 ArcFace model")
    print("  2) Quantize to INT8 pack")
    answer = input("Run setup now? (y/n): ").strip().lower()
    if answer not in {"y", "yes"}:
        return False

    rc = run_python_script("src/arcface/setup_model.py", [], "ArcFace setup model")
    if rc != 0:
        return False

    rc = run_python_script("src/arcface_mobilenet_int8/quantize_model.py", [], "ArcFace INT8 quantize model")
    if rc != 0:
        return False

    return int8_model_pack_ready()


def print_model_menu() -> None:
    print("\nChoose a model/type:")
    for idx, (model_name, _) in enumerate(GROUPED_CHOICES, start=1):
        print(f"{idx:2d}. {model_name}")
    print(" q. Quit")


def print_model_actions_menu(model_name: str, actions: list[tuple[str, str]]) -> None:
    info = get_model_info(model_name)
    trained_state = "trained" if info["trained"] else "untrained"
    eval_state = "evaluated" if info["evaluated"] else "not evaluated"

    print(f"\n[{model_name}]")
    print(f"State: {trained_state}")
    print(f"Evaluation: {eval_state}")
    print(f"Size: {info['size']}")
    if model_name == "Benchmark":
        print_benchmark_overview()
    print("\nChoices:")
    for idx, (action_label, _) in enumerate(actions, start=1):
        print(f"{idx:2d}. {action_label}")
    print(" b. Back")
    print(" q. Quit")


def run_choice(model_name: str, action_label: str, rel_script: str, extra_args: list[str]) -> int:
    script_path = resolve_path(rel_script)
    cmd = [*get_python_command(), str(script_path), *extra_args]

    print(f"\n[RUN] {model_name}: {action_label}")
    print(f"[CMD] {' '.join(shlex.quote(part) for part in cmd)}\n")

    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    print(f"\n[EXIT] code={completed.returncode}")
    return completed.returncode


def main() -> int:
    while True:
        print_model_menu()
        selected_model = input("\nEnter model number (or q): ").strip().lower()

        if selected_model in {"q", "quit", "exit"}:
            print("Exiting.")
            return 0

        if not selected_model.isdigit():
            print("Invalid input. Enter a model number or q.")
            continue

        model_index = int(selected_model) - 1
        if model_index < 0 or model_index >= len(GROUPED_CHOICES):
            print("Invalid model number.")
            continue

        model_name, actions = GROUPED_CHOICES[model_index]

        while True:
            print_model_actions_menu(model_name, actions)
            selected_action = input("\nEnter choice number (or b/q): ").strip().lower()

            if selected_action in {"q", "quit", "exit"}:
                print("Exiting.")
                return 0
            if selected_action in {"b", "back"}:
                break
            if not selected_action.isdigit():
                print("Invalid input. Enter an action number, b, or q.")
                continue

            action_index = int(selected_action) - 1
            if action_index < 0 or action_index >= len(actions):
                print("Invalid action number.")
                continue

            action_label, rel_script = actions[action_index]
            if (
                model_name == "ArcFace MobileNet INT8"
                and action_label in {"train enrollment", "evaluate", "live detect"}
                and not int8_model_pack_ready()
            ):
                if not maybe_prepare_int8_pack():
                    print("[INFO] Skipping action because INT8 model pack is not ready.")
                    continue
            extra = input("Optional extra args (or press Enter): ").strip()
            extra_args = shlex.split(extra) if extra else []
            run_choice(model_name, action_label, rel_script, extra_args)

            again = input("\nRun another action for this model? (y/n): ").strip().lower()
            if again not in {"y", "yes"}:
                break


if __name__ == "__main__":
    raise SystemExit(main())
