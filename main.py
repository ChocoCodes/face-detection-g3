import shlex
import subprocess
import json
import os
import hashlib
import re
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
        "Eigenfaces",
        [
            ("train", "src/eigenfaces/trainer.py"),
            ("evaluate", "src/eigenfaces/evaluate.py"),
            ("live detect", "src/eigenfaces/detect.py"),
        ],
    ),
    (
        "Fisherfaces",
        [
            ("train", "src/fisherfaces/trainer.py"),
            ("evaluate", "src/fisherfaces/evaluate.py"),
            ("live detect", "src/fisherfaces/detect.py"),
        ],
    ),
    (
        "Benchmark",
        [
            ("compare models", "src/benchmark/compare_models.py"),
            ("aggregate live FPS", "src/benchmark/aggregate_live_fps.py"),
            ("aggregate evaluation reports", "src/benchmark/aggregate_evaluation_reports.py"),
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
    "Eigenfaces": {
        "trained_markers": [
            "models/eigenfaces/trainer_eigenfaces.yml",
            "models/eigenfaces/labels_eigenfaces.json",
        ],
        "evaluated_reports": ["reports/evaluation/eigenfaces_eval.json"],
        "size_paths": [
            "models/eigenfaces/trainer_eigenfaces.yml",
            "models/eigenfaces/labels_eigenfaces.json",
        ],
    },
    "Fisherfaces": {
        "trained_markers": [
            "models/fisherfaces/trainer_fisherfaces.yml",
            "models/fisherfaces/labels_fisherfaces.json",
        ],
        "evaluated_reports": ["reports/evaluation/fisherfaces_eval.json"],
        "size_paths": [
            "models/fisherfaces/trainer_fisherfaces.yml",
            "models/fisherfaces/labels_fisherfaces.json",
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
    "Eigenfaces": {
        "eval_report": "reports/evaluation/eigenfaces_eval.json",
        "fps_algorithm": "eigenfaces",
    },
    "Fisherfaces": {
        "eval_report": "reports/evaluation/fisherfaces_eval.json",
        "fps_algorithm": "fisherfaces",
    },
}

MODEL_FAMILY_ALIASES: dict[str, set[str]] = {
    "ArcFace": {"arcface_buffalo_s"},
    "ArcFace MobileNet INT8": {"arcface_mobilenet_int8", "arcface_buffalo_s_int8"},
    "MobileFaceNet": {"yunet_mobilefacenet"},
    "EdgeFace": {"edgeface"},
    "LBPH": {"lbph"},
    "Eigenfaces": {"eigenfaces"},
    "Fisherfaces": {"fisherfaces"},
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
    overall = report_payload.get("overall", {})
    if "hit_rate_percent" in overall:
        try:
            return float(overall["hit_rate_percent"])
        except Exception:
            pass
    if "correct" in overall and "evaluated_images" in overall:
        try:
            correct = float(overall["correct"])
            evaluated = float(overall["evaluated_images"])
            if evaluated > 0:
                return 100.0 * correct / evaluated
        except Exception:
            pass
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
    return None


def extract_accuracy_percent(report_payload: dict | None) -> float | None:
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
    return extract_hit_rate_percent(report_payload)


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


def collect_evaluation_entities() -> list[dict]:
    reports_dir = resolve_path("reports/evaluation")
    if not reports_dir.exists():
        return []

    rows: list[dict] = []
    for report_path in sorted(reports_dir.glob("*.json")):
        payload = load_json_if_exists(report_path)
        if not payload:
            continue
        model_family = str(payload.get("model_family", "")).strip()
        if not model_family:
            continue

        hit_rate = extract_hit_rate_percent(payload)
        overall = payload.get("overall", {}) if isinstance(payload.get("overall"), dict) else {}
        evaluated_images = int(overall.get("evaluated_images", 0)) if overall else 0
        dataset_profile = payload.get("dataset_profile", {})
        dataset_label = "unknown"
        if isinstance(dataset_profile, dict):
            dataset_label = str(dataset_profile.get("label", "unknown"))

        rows.append(
            {
                "model_family": model_family,
                "model_variant": str(payload.get("model_variant", "default")),
                "entity_key": str(payload.get("entity_key", report_path.stem)),
                "dataset_label": dataset_label,
                "hit_rate": hit_rate,
                "evaluated_images": evaluated_images,
            }
        )

    rows.sort(
        key=lambda row: (
            row["hit_rate"] if row["hit_rate"] is not None else float("-inf"),
            row["evaluated_images"],
        ),
        reverse=True,
    )
    return rows


def entities_for_menu_model(model_name: str) -> list[dict]:
    aliases = {value.lower() for value in MODEL_FAMILY_ALIASES.get(model_name, set())}
    if not aliases:
        return []
    rows = collect_evaluation_entities()
    return [row for row in rows if str(row.get("model_family", "")).strip().lower() in aliases]


def get_arg_value(args: list[str], flag: str, default: str) -> str:
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return default


def bool_flag(args: list[str], name: str, default: bool) -> bool:
    positive = f"--{name}"
    negative = f"--no-{name}"
    if any(arg == negative or arg.startswith(f"{negative}=") for arg in args):
        return False
    if any(arg == positive or arg.startswith(f"{positive}=") for arg in args):
        return True
    return default


def infer_split_name(raw_dir_name: str, processed_dir_name: str) -> str:
    raw_base = Path(raw_dir_name).name.lower()
    processed_base = Path(processed_dir_name).name.lower()
    for candidate in (processed_base, raw_base):
        if candidate in {"train", "test"}:
            return candidate
    return ""


def build_dataset_label_from_args(
    args: list[str],
    *,
    is_training: bool,
    is_evaluation: bool,
) -> str:
    raw_dir_name = get_arg_value(args, "--raw-dir-name", "lasalle_db1")
    processed_dir_name = get_arg_value(args, "--processed-dir-name", "lfw-dataset")
    augmented_dir_name = get_arg_value(args, "--augmented-dir-name", "split_augmented41mods")
    aug_splits_raw = get_arg_value(args, "--aug-splits", "__disabled__")

    include_raw = bool_flag(args, "include-raw", raw_dir_name != "__disabled__")
    include_processed = bool_flag(args, "include-processed", False)
    include_augmented_default = True if is_training and not is_evaluation else False
    include_augmented = bool_flag(args, "include-augmented", include_augmented_default)

    if raw_dir_name == "__disabled__":
        include_raw = False
    if aug_splits_raw.strip() == "__disabled__":
        include_augmented = False

    aug_splits = [
        value.strip().lower()
        for value in aug_splits_raw.split(",")
        if value.strip() and value.strip() != "__disabled__"
    ]
    joined_aug = ",".join(sorted(set(aug_splits))) if aug_splits else "all"

    tokens: list[str] = []
    if include_raw:
        tokens.append(f"raw={raw_dir_name}")
    if include_processed:
        tokens.append(f"processed={processed_dir_name}")
    if include_augmented:
        tokens.append(f"aug={augmented_dir_name}[{joined_aug}]")

    split_name = infer_split_name(raw_dir_name=raw_dir_name, processed_dir_name=processed_dir_name)
    if split_name:
        tokens.append(f"split={split_name}")

    return " | ".join(tokens) if tokens else "no_dataset_selected"


def combo_slug_for_args(
    args: list[str],
    *,
    is_training: bool,
    is_evaluation: bool,
) -> str:
    label = build_dataset_label_from_args(
        args,
        is_training=is_training,
        is_evaluation=is_evaluation,
    )
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", label.lower()).strip("_")
    if not clean:
        clean = "dataset"
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
    compact = clean[:48].strip("_")
    if not compact:
        compact = "dataset"
    return f"{compact}_{digest}"


def auto_artifact_args_for_action(
    *,
    model_name: str,
    rel_script: str,
    base_args: list[str],
    is_training: bool,
    is_evaluation: bool,
) -> list[str]:
    # Respect explicit user-provided artifact paths.
    if model_name == "LBPH":
        if is_training:
            if has_flag(base_args, "--model-output") or has_flag(base_args, "--labels-output"):
                return []
            slug = combo_slug_for_args(base_args, is_training=True, is_evaluation=False)
            return [
                "--model-output",
                f"models/lbph/trainer_{slug}.yml",
                "--labels-output",
                f"models/lbph/labels_{slug}.json",
            ]
        if is_evaluation:
            if has_flag(base_args, "--model-path") or has_flag(base_args, "--labels-path"):
                return []
            slug = combo_slug_for_args(base_args, is_training=False, is_evaluation=True)
            return [
                "--model-path",
                f"models/lbph/trainer_{slug}.yml",
                "--labels-path",
                f"models/lbph/labels_{slug}.json",
            ]

    if model_name == "Eigenfaces":
        if is_training:
            if has_flag(base_args, "--model-output") or has_flag(base_args, "--labels-output"):
                return []
            slug = combo_slug_for_args(base_args, is_training=True, is_evaluation=False)
            return [
                "--model-output",
                f"models/eigenfaces/trainer_{slug}.yml",
                "--labels-output",
                f"models/eigenfaces/labels_{slug}.json",
            ]
        if is_evaluation:
            if has_flag(base_args, "--model-path") or has_flag(base_args, "--labels-path"):
                return []
            slug = combo_slug_for_args(base_args, is_training=False, is_evaluation=True)
            return [
                "--model-path",
                f"models/eigenfaces/trainer_{slug}.yml",
                "--labels-path",
                f"models/eigenfaces/labels_{slug}.json",
            ]

    if model_name == "Fisherfaces":
        if is_training:
            if has_flag(base_args, "--model-output") or has_flag(base_args, "--labels-output"):
                return []
            slug = combo_slug_for_args(base_args, is_training=True, is_evaluation=False)
            return [
                "--model-output",
                f"models/fisherfaces/trainer_{slug}.yml",
                "--labels-output",
                f"models/fisherfaces/labels_{slug}.json",
            ]
        if is_evaluation:
            if has_flag(base_args, "--model-path") or has_flag(base_args, "--labels-path"):
                return []
            slug = combo_slug_for_args(base_args, is_training=False, is_evaluation=True)
            return [
                "--model-path",
                f"models/fisherfaces/trainer_{slug}.yml",
                "--labels-path",
                f"models/fisherfaces/labels_{slug}.json",
            ]

    if model_name in {"ArcFace", "ArcFace MobileNet INT8"}:
        if is_training and not has_flag(base_args, "--enrollment-output"):
            slug = combo_slug_for_args(base_args, is_training=True, is_evaluation=False)
            return [
                "--enrollment-output",
                f"models/arcface_mobilenet/enrollment_{slug}.json",
            ]
        if is_evaluation and not has_flag(base_args, "--enrollment-path"):
            slug = combo_slug_for_args(base_args, is_training=False, is_evaluation=True)
            return [
                "--enrollment-path",
                f"models/arcface_mobilenet/enrollment_{slug}.json",
            ]

    if model_name == "MobileFaceNet":
        if is_training and not has_flag(base_args, "--enrollment-output"):
            slug = combo_slug_for_args(base_args, is_training=True, is_evaluation=False)
            return [
                "--enrollment-output",
                f"models/yunet_mobilefacenet/enrollment_{slug}.json",
            ]
        if is_evaluation and not has_flag(base_args, "--enrollment-path"):
            slug = combo_slug_for_args(base_args, is_training=False, is_evaluation=True)
            return [
                "--enrollment-path",
                f"models/yunet_mobilefacenet/enrollment_{slug}.json",
            ]

    if model_name == "EdgeFace":
        if is_training and not has_flag(base_args, "--enrollment-output"):
            slug = combo_slug_for_args(base_args, is_training=True, is_evaluation=False)
            return [
                "--enrollment-output",
                f"models/edgeface/enrollment_{slug}.json",
            ]
        if is_evaluation and not has_flag(base_args, "--enrollment-path"):
            slug = combo_slug_for_args(base_args, is_training=False, is_evaluation=True)
            return [
                "--enrollment-path",
                f"models/edgeface/enrollment_{slug}.json",
            ]

    return []


def warn_if_missing_auto_artifacts(args: list[str], is_evaluation: bool) -> None:
    if not is_evaluation:
        return
    path_flags = ["--model-path", "--labels-path", "--enrollment-path"]
    checked_any = False
    for flag in path_flags:
        if not has_flag(args, flag):
            continue
        value = get_arg_value(args, flag, "").strip()
        if not value:
            continue
        checked_any = True
        if not resolve_path(value).exists():
            print(f"[WARN] Selected artifact does not exist yet: {value}")
    if checked_any:
        print("[INFO] If needed, override with Optional extra args.")


def maybe_confirm_existing_dataset_combo(
    *,
    model_name: str,
    final_args: list[str],
    is_training: bool,
    is_evaluation: bool,
) -> bool:
    model_entities = entities_for_menu_model(model_name)
    if not model_entities:
        return True

    selected_label = build_dataset_label_from_args(
        final_args,
        is_training=is_training,
        is_evaluation=is_evaluation,
    )
    matches = [row for row in model_entities if row.get("dataset_label", "") == selected_label]
    if not matches:
        return True

    print("\n[INFO] Existing dataset combination detected for this model:")
    print(f"  selected: {selected_label}")
    for row in matches[:5]:
        hit_rate_display = (
            f"{row['hit_rate']:.2f}%"
            if row.get("hit_rate") is not None
            else "N/A"
        )
        print(
            f"  - variant={row.get('model_variant', 'default')} "
            f"| hit={hit_rate_display} "
            f"| eval={row.get('evaluated_images', 0)} "
            f"| entity={row.get('entity_key', 'n/a')}"
        )

    answer = input("Continue anyway? (y/n, default n): ").strip().lower()
    return answer in {"y", "yes"}


def print_benchmark_overview() -> None:
    fps_by_algo = collect_fps_summary()
    rows: list[dict] = []

    for model_name, cfg in BENCHMARK_OVERVIEW_CONFIG.items():
        report_path = resolve_path(cfg["eval_report"])
        report_payload = load_json_if_exists(report_path)
        hit_rate = extract_hit_rate_percent(report_payload)
        accuracy = extract_accuracy_percent(report_payload)

        algo_key = str(cfg["fps_algorithm"]).strip().lower()
        fps_value = fps_by_algo.get(algo_key)
        size_display = get_model_info(model_name)["size"]

        rows.append(
            {
                "model": model_name,
                "hit_rate": hit_rate,
                "accuracy": accuracy,
                "fps": fps_value,
                "size": size_display,
            }
        )

    # Sort by highest hit rate first, then highest accuracy.
    rows.sort(
        key=lambda row: (
            row["hit_rate"] if row["hit_rate"] is not None else float("-inf"),
            row["accuracy"] if row["accuracy"] is not None else float("-inf"),
        ),
        reverse=True,
    )

    print("\nOverview (all models)")
    print(f"{'Model':<28} {'Hit Rate':>10} {'Accuracy':>10} {'Avg FPS':>10} {'Size':>12}")
    print(f"{'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for row in rows:
        hit_rate_display = f"{row['hit_rate']:.2f}%" if row["hit_rate"] is not None else "N/A"
        accuracy_display = f"{row['accuracy']:.2f}%" if row["accuracy"] is not None else "N/A"
        fps_display = f"{row['fps']:.2f}" if row["fps"] is not None else "N/A"
        print(
            f"{row['model']:<28} {hit_rate_display:>10} {accuracy_display:>10} "
            f"{fps_display:>10} {row['size']:>12}"
        )

    entity_rows = collect_evaluation_entities()
    if entity_rows:
        print("\nEvaluation entities (same model + different dataset/split are separate)")
        print(f"{'Model Family':<24} {'Variant':<28} {'Hit Rate':>10} {'Eval':>8} {'Dataset':<64}")
        print(f"{'-'*24} {'-'*28} {'-'*10} {'-'*8} {'-'*64}")
        for row in entity_rows[:20]:
            hit_rate_display = f"{row['hit_rate']:.2f}%" if row["hit_rate"] is not None else "N/A"
            dataset_display = row["dataset_label"]
            if len(dataset_display) > 64:
                dataset_display = dataset_display[:61] + "..."
            print(
                f"{row['model_family']:<24} {row['model_variant']:<28} "
                f"{hit_rate_display:>10} {row['evaluated_images']:>8} {dataset_display:<64}"
            )


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


def has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def is_training_action(action_label: str, rel_script: str) -> bool:
    return rel_script.endswith("trainer.py") and action_label.startswith("train")


def is_evaluate_action(action_label: str, rel_script: str) -> bool:
    return rel_script.endswith("evaluate.py") and action_label.startswith("evaluate")


def remove_flag_and_value(args: list[str], flag: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for idx, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            if idx + 1 < len(args):
                skip_next = True
            continue
        if arg.startswith(f"{flag}="):
            continue
        out.append(arg)
    return out


def prompt_core_dataset_args(is_training: bool) -> list[str]:
    phase_label = "training" if is_training else "evaluation"

    print(f"\nSelect base dataset source for {phase_label}:")
    print("  1. La Salle only -> data/lasalle_db1")
    print("  2. LFW only      -> data/lfw-dataset")
    print("  3. Both          -> data/lasalle_db1 + data/lfw-dataset")
    selected = input("Enter choice (default: 1): ").strip()

    include_raw = True
    include_processed = False
    if selected in {"", "1"}:
        include_raw = True
        include_processed = False
    elif selected == "2":
        include_raw = False
        include_processed = True
    elif selected == "3":
        include_raw = True
        include_processed = True
    else:
        print("[INFO] Invalid choice; defaulting to La Salle only.")

    args: list[str] = [
        "--base-data-dir",
        "data",
        "--raw-dir-name",
        "lasalle_db1" if include_raw else "__disabled__",
        "--processed-dir-name",
        "lfw-dataset",
    ]

    if is_training:
        if include_raw:
            args.append("--include-raw")
    if include_processed:
        args.append("--include-processed")

    return args


def prompt_augmented_dataset_args(is_evaluation: bool) -> list[str]:
    print("\nSelect augmented datasets (comma/space separated numbers):")
    print("  1. light   -> data/split_augmented41mods/light")
    print("  2. medium  -> data/split_augmented41mods/medium")
    selected = input("Enter choices (blank = none): ").strip()

    if not selected:
        chosen: list[str] = []
    else:
        tokens = selected.replace(",", " ").split()
        mapping = {"1": "light", "2": "medium"}
        chosen = []
        for token in tokens:
            value = mapping.get(token)
            if value is None:
                print(f"[WARN] Ignoring invalid choice: {token}")
                continue
            if value not in chosen:
                chosen.append(value)

    args: list[str] = [
        "--augmented-dir-name",
        "split_augmented41mods",
    ]
    if chosen:
        args.extend(["--aug-splits", ",".join(chosen)])
        if is_evaluation:
            args.append("--include-augmented")
    else:
        args.extend(["--aug-splits", "__disabled__"])
        if is_evaluation:
            args.append("--no-include-augmented")

    return args


def run_python_script(rel_script: str, extra_args: list[str], label: str) -> int:
    script_path = resolve_path(rel_script)
    cmd = [*get_python_command(), str(script_path), *extra_args]
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    project_root_str = str(PROJECT_ROOT)
    env["PYTHONPATH"] = (
        f"{project_root_str}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else project_root_str
    )
    print(f"\n[RUN] {label}")
    print(f"[CMD] {' '.join(shlex.quote(part) for part in cmd)}\n")
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
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
    model_entities = entities_for_menu_model(model_name)
    if model_entities:
        print(f"Saved dataset combos: {len(model_entities)}")
        for row in model_entities[:3]:
            hit_rate_display = (
                f"{row['hit_rate']:.2f}%"
                if row.get("hit_rate") is not None
                else "N/A"
            )
            print(
                f"  - {row.get('dataset_label', 'unknown')} "
                f"(variant={row.get('model_variant', 'default')}, hit={hit_rate_display})"
            )
    else:
        print("Saved dataset combos: 0")
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
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    project_root_str = str(PROJECT_ROOT)
    env["PYTHONPATH"] = (
        f"{project_root_str}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else project_root_str
    )

    print(f"\n[RUN] {model_name}: {action_label}")
    print(f"[CMD] {' '.join(shlex.quote(part) for part in cmd)}\n")

    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
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
            training_action = is_training_action(action_label, rel_script)
            evaluate_action = is_evaluate_action(action_label, rel_script)
            if (
                model_name == "ArcFace MobileNet INT8"
                and action_label in {"train enrollment", "evaluate", "live detect"}
                and not int8_model_pack_ready()
            ):
                if not maybe_prepare_int8_pack():
                    print("[INFO] Skipping action because INT8 model pack is not ready.")
                    continue

            preset_args: list[str] = []
            if training_action or evaluate_action:
                preset_args = [
                    *prompt_core_dataset_args(is_training=training_action),
                    *prompt_augmented_dataset_args(is_evaluation=evaluate_action),
                ]

            extra = input("Optional extra args (or press Enter): ").strip()
            extra_args = shlex.split(extra) if extra else []
            if has_flag(extra_args, "--include-raw") or has_flag(extra_args, "--no-include-raw"):
                preset_args = [arg for arg in preset_args if arg != "--include-raw"]
            if has_flag(extra_args, "--include-processed") or has_flag(extra_args, "--no-include-processed"):
                preset_args = [arg for arg in preset_args if arg != "--include-processed"]
            if has_flag(extra_args, "--include-augmented") or has_flag(extra_args, "--no-include-augmented"):
                preset_args = [arg for arg in preset_args if arg not in {"--include-augmented", "--no-include-augmented"}]
            if has_flag(extra_args, "--base-data-dir"):
                preset_args = remove_flag_and_value(preset_args, "--base-data-dir")
            if has_flag(extra_args, "--raw-dir-name"):
                preset_args = remove_flag_and_value(preset_args, "--raw-dir-name")
            if has_flag(extra_args, "--processed-dir-name"):
                preset_args = remove_flag_and_value(preset_args, "--processed-dir-name")
            if has_flag(extra_args, "--augmented-dir-name"):
                preset_args = remove_flag_and_value(preset_args, "--augmented-dir-name")
            if has_flag(extra_args, "--aug-splits"):
                preset_args = remove_flag_and_value(preset_args, "--aug-splits")

            base_args = [*preset_args, *extra_args]
            auto_args = auto_artifact_args_for_action(
                model_name=model_name,
                rel_script=rel_script,
                base_args=base_args,
                is_training=training_action,
                is_evaluation=evaluate_action,
            )
            final_args = [*base_args, *auto_args]
            warn_if_missing_auto_artifacts(final_args, is_evaluation=evaluate_action)
            if training_action or evaluate_action:
                should_continue = maybe_confirm_existing_dataset_combo(
                    model_name=model_name,
                    final_args=final_args,
                    is_training=training_action,
                    is_evaluation=evaluate_action,
                )
                if not should_continue:
                    print("[INFO] Action cancelled. Choose another dataset combination or add custom args.")
                    continue
            run_choice(model_name, action_label, rel_script, final_args)

            again = input("\nRun another action for this model? (y/n): ").strip().lower()
            if again not in {"y", "yes"}:
                break


if __name__ == "__main__":
    raise SystemExit(main())
