import argparse
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def root_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT.joinpath(candidate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create INT8 ArcFace model pack by quantizing w600k_mbf.onnx."
    )
    parser.add_argument(
        "--source-model-dir",
        default=str(root_path("models", "arcface_mobilenet")),
        help="Source ArcFace model directory (FP32).",
    )
    parser.add_argument(
        "--target-model-dir",
        default=str(root_path("models", "arcface_mobilenet_int8")),
        help="Target ArcFace model directory (INT8).",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Do not copy source tree; quantize directly in existing target tree.",
    )
    parser.add_argument(
        "--weight-type",
        choices=["qint8", "quint8"],
        default="qint8",
        help="Quantized weight data type.",
    )
    return parser.parse_args()


def quantize_file(src: Path, dst: Path, weight_type: QuantType) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=weight_type,
        per_channel=True,
    )
    print(f"[OK] Quantized: {dst}")


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_model_dir)
    target_dir = resolve_path(args.target_model_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source model dir not found: {source_dir}")

    if not args.skip_copy:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        print(f"[OK] Copied model tree: {source_dir} -> {target_dir}")
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    qt = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8

    root_model = source_dir / "w600k_mbf.onnx"
    root_model_out = target_dir / "w600k_mbf.onnx"
    quantize_file(root_model, root_model_out, qt)

    zoo_model = source_dir / "models" / "buffalo_s" / "w600k_mbf.onnx"
    zoo_model_out = target_dir / "models" / "buffalo_s" / "w600k_mbf.onnx"
    quantize_file(zoo_model, zoo_model_out, qt)

    print(f"[DONE] INT8 ArcFace pack ready at: {target_dir}")


if __name__ == "__main__":
    main()

