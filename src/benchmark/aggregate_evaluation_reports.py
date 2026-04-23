import argparse
import json
from pathlib import Path
from typing import Any

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
        description="Aggregate evaluation reports by model+dataset entity key."
    )
    parser.add_argument(
        "--reports-dir",
        default=root_path("reports", "evaluation"),
        help="Directory containing evaluator JSON reports.",
    )
    parser.add_argument(
        "--output-json",
        default=root_path("reports", "benchmark", "evaluation_entities_summary.json"),
        help="Path to write aggregated JSON summary.",
    )
    parser.add_argument(
        "--output-md",
        default=root_path("reports", "benchmark", "evaluation_entities_summary.md"),
        help="Path to write aggregated Markdown summary.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception:
        return None


def metric_hit_rate(report: dict[str, Any]) -> float:
    overall = report.get("overall", {})
    if isinstance(overall, dict):
        value = overall.get("hit_rate_percent")
        if isinstance(value, (int, float)):
            return float(value)
        correct = overall.get("correct")
        total = overall.get("evaluated_images")
        if isinstance(correct, (int, float)) and isinstance(total, (int, float)) and total:
            return (100.0 * float(correct)) / float(total)
    return 0.0


def metric_evaluated(report: dict[str, Any]) -> int:
    overall = report.get("overall", {})
    if isinstance(overall, dict):
        value = overall.get("evaluated_images")
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def to_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Evaluation Entities Summary")
    lines.append("")
    lines.append(f"- Reports scanned: {summary['reports_scanned']}")
    lines.append(f"- Entities: {summary['entities_count']}")
    lines.append("")
    lines.append("| Model Family | Variant | Dataset | Hit Rate (%) | Evaluated | Runs | Latest Report |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for row in summary["entities"]:
        lines.append(
            f"| {row['model_family']} | {row['model_variant']} | {row['dataset_label']} | "
            f"{row['overall_hit_rate_percent']:.2f} | {row['evaluated_images']} | "
            f"{row['runs']} | {row['latest_report']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.reports_dir = resolve_path(args.reports_dir)
    args.output_json = resolve_path(args.output_json)
    args.output_md = resolve_path(args.output_md)

    reports_dir = Path(args.reports_dir)
    report_files = sorted(reports_dir.glob("*.json"))
    if not report_files:
        raise RuntimeError(f"No JSON reports found in: {reports_dir}")

    grouped: dict[str, list[dict[str, Any]]] = {}

    for file_path in report_files:
        payload = read_json(file_path)
        if not payload:
            continue
        model_family = str(payload.get("model_family", "")).strip()
        if not model_family:
            continue

        entity_key = str(payload.get("entity_key", "")).strip()
        if not entity_key:
            entity_key = f"{model_family}::{file_path.stem}"

        grouped.setdefault(entity_key, []).append(
            {
                "path": str(file_path),
                "mtime": file_path.stat().st_mtime,
                "payload": payload,
            }
        )

    rows: list[dict[str, Any]] = []
    for entity_key, reports in sorted(grouped.items()):
        reports.sort(key=lambda row: row["mtime"], reverse=True)
        latest = reports[0]
        payload = latest["payload"]
        dataset_profile = payload.get("dataset_profile", {})
        dataset_label = "unknown"
        if isinstance(dataset_profile, dict):
            dataset_label = str(dataset_profile.get("label", "unknown"))

        rows.append(
            {
                "entity_key": entity_key,
                "model_family": str(payload.get("model_family", "unknown")),
                "model_variant": str(payload.get("model_variant", "default")),
                "dataset_key": str(payload.get("dataset_key", "")),
                "dataset_label": dataset_label,
                "overall_hit_rate_percent": metric_hit_rate(payload),
                "evaluated_images": metric_evaluated(payload),
                "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
                "runs": len(reports),
                "latest_report": str(Path(str(latest["path"])).relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "report_paths": [
                    str(Path(str(row["path"])).relative_to(PROJECT_ROOT)).replace("\\", "/")
                    for row in reports
                ],
            }
        )

    rows.sort(
        key=lambda row: (row["overall_hit_rate_percent"], row["evaluated_images"], row["model_family"]),
        reverse=True,
    )

    summary = {
        "reports_dir": str(reports_dir),
        "reports_scanned": len(report_files),
        "entities_count": len(rows),
        "entities": rows,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with out_md.open("w", encoding="utf-8") as f:
        f.write(to_markdown(summary))

    print("[OK] Evaluation entities summary written")
    print(f"  - JSON: {out_json}")
    print(f"  - Markdown: {out_md}")


if __name__ == "__main__":
    main()
