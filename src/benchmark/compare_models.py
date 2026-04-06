import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def root_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def resolve_path(path_value: str) -> str:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)
    return str(PROJECT_ROOT.joinpath(candidate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and compare LBPH vs YuNet+MobileFaceNet evaluators."
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick: smaller sample size, full: full dataset run.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--lbph-report",
        default=root_path("reports", "benchmark", "lbph_eval.json"),
    )
    parser.add_argument(
        "--embedding-report",
        default=root_path("reports", "benchmark", "yunet_mobilefacenet_eval.json"),
    )
    parser.add_argument(
        "--summary-json",
        default=root_path("reports", "benchmark", "comparison_summary.json"),
    )
    parser.add_argument(
        "--summary-md",
        default=root_path("reports", "benchmark", "comparison_summary.md"),
    )
    parser.add_argument("--unknown-threshold-lbph", type=float, default=55.0)
    parser.add_argument("--embedding-threshold", type=float, default=-1.0)
    parser.add_argument("--assume-processed-are-cropped", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], label: str) -> None:
    print(f"[RUN] {label}")
    print("      " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def by_bucket(report: dict) -> dict:
    return {row["bucket"]: row for row in report.get("buckets", [])}


def hit_rate(report: dict) -> float:
    return float(report.get("overall", {}).get("hit_rate_percent", 0.0))


def elapsed(report: dict) -> float:
    return float(report.get("elapsed_seconds", 0.0))


def build_markdown(summary: dict) -> str:
    lines = []
    lines.append("# Model Comparison Summary")
    lines.append("")
    lines.append(f"- Generated UTC: {summary['generated_at_utc']}")
    lines.append(f"- Mode: {summary['mode']}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Model | Hit Rate (%) | Eval Time (s) |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| LBPH | {summary['lbph']['overall_hit_rate_percent']:.2f} | {summary['lbph']['elapsed_seconds']:.2f} |"
    )
    lines.append(
        f"| YuNet+MobileFaceNet | {summary['embedding']['overall_hit_rate_percent']:.2f} | {summary['embedding']['elapsed_seconds']:.2f} |"
    )
    lines.append("")
    lines.append("## Per Bucket Hit Rate")
    lines.append("")
    lines.append("| Bucket | LBPH (%) | YuNet+MobileFaceNet (%) | Delta (Emb-LBPH) |")
    lines.append("|---|---:|---:|---:|")

    for row in summary["bucket_comparison"]:
        lines.append(
            f"| {row['bucket']} | {row['lbph_hit_rate_percent']:.2f} | {row['embedding_hit_rate_percent']:.2f} | {row['delta_percent']:+.2f} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Higher hit rate means better identity prediction accuracy.")
    lines.append("- Report files include misclassification samples for debugging.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.lbph_report = resolve_path(args.lbph_report)
    args.embedding_report = resolve_path(args.embedding_report)
    args.summary_json = resolve_path(args.summary_json)
    args.summary_md = resolve_path(args.summary_md)

    os.makedirs(os.path.dirname(args.lbph_report), exist_ok=True)
    os.makedirs(os.path.dirname(args.embedding_report), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_md), exist_ok=True)

    max_images = "5" if args.mode == "quick" else "0"
    aug_splits = "original,light" if args.mode == "quick" else "original,light,medium,heavy"

    lbph_cmd = [
        args.python,
        root_path("src", "lbph", "evaluate.py"),
        "--max-images-per-person",
        max_images,
        "--aug-splits",
        aug_splits,
        "--unknown-threshold",
        str(args.unknown_threshold_lbph),
        "--report-json",
        args.lbph_report,
    ]

    embedding_cmd = [
        args.python,
        root_path("src", "yunet_mobilefacenet", "evaluate.py"),
        "--max-images-per-person",
        max_images,
        "--aug-splits",
        aug_splits,
        "--report-json",
        args.embedding_report,
    ]
    if args.embedding_threshold >= 0:
        embedding_cmd.extend(["--threshold", str(args.embedding_threshold)])

    if args.assume_processed_are_cropped:
        lbph_cmd.extend(["--assume-processed-are-cropped", "--downscale-max-side", "384"])
        embedding_cmd.append("--assume-processed-are-cropped")

    run_command(lbph_cmd, "LBPH evaluator")
    run_command(embedding_cmd, "YuNet+MobileFaceNet evaluator")

    with open(args.lbph_report, "r", encoding="utf-8") as f:
        lbph_report = json.load(f)
    with open(args.embedding_report, "r", encoding="utf-8") as f:
        embedding_report = json.load(f)

    lbph_buckets = by_bucket(lbph_report)
    emb_buckets = by_bucket(embedding_report)
    all_buckets = sorted(set(lbph_buckets.keys()) | set(emb_buckets.keys()))

    bucket_rows = []
    for bucket in all_buckets:
        lbph_hit = float(lbph_buckets.get(bucket, {}).get("hit_rate_percent", 0.0))
        emb_hit = float(emb_buckets.get(bucket, {}).get("hit_rate_percent", 0.0))
        bucket_rows.append(
            {
                "bucket": bucket,
                "lbph_hit_rate_percent": lbph_hit,
                "embedding_hit_rate_percent": emb_hit,
                "delta_percent": emb_hit - lbph_hit,
            }
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "lbph": {
            "report_path": args.lbph_report,
            "overall_hit_rate_percent": hit_rate(lbph_report),
            "elapsed_seconds": elapsed(lbph_report),
        },
        "embedding": {
            "report_path": args.embedding_report,
            "overall_hit_rate_percent": hit_rate(embedding_report),
            "elapsed_seconds": elapsed(embedding_report),
        },
        "bucket_comparison": bucket_rows,
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(args.summary_md, "w", encoding="utf-8") as f:
        f.write(build_markdown(summary))

    print("[OK] Benchmark summary written")
    print(f"  - JSON: {args.summary_json}")
    print(f"  - Markdown: {args.summary_md}")


if __name__ == "__main__":
    main()
