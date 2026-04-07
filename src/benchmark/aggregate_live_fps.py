import argparse
import json
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
        description="Aggregate average FPS across live detect.py runs."
    )
    parser.add_argument(
        "--runs-dir",
        default=root_path("reports", "benchmark", "live_fps", "runs"),
        help="Directory containing per-run FPS summary JSON files.",
    )
    parser.add_argument(
        "--output-json",
        default=root_path("reports", "benchmark", "live_fps", "aggregate_summary.json"),
        help="Path to write aggregate FPS summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.runs_dir = resolve_path(args.runs_dir)
    args.output_json = resolve_path(args.output_json)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    run_files = sorted(runs_dir.glob("*.json"))
    if not run_files:
        raise RuntimeError(f"No run summaries found in: {runs_dir}")

    grouped: dict[str, list[dict]] = {}
    for path in run_files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        algo = str(payload.get("algorithm", "unknown"))
        grouped.setdefault(algo, []).append(payload)

    aggregate_rows = []
    for algo, rows in sorted(grouped.items()):
        total_frames = sum(int(r.get("frames", 0)) for r in rows)
        total_duration = sum(float(r.get("duration_seconds", 0.0)) for r in rows)

        if total_duration > 0 and total_frames > 0:
            weighted_avg_fps = total_frames / total_duration
        else:
            fps_values = [float(r.get("average_fps", 0.0)) for r in rows]
            weighted_avg_fps = (sum(fps_values) / len(fps_values)) if fps_values else 0.0

        aggregate_rows.append(
            {
                "algorithm": algo,
                "runs": len(rows),
                "total_frames": total_frames,
                "total_duration_seconds": total_duration,
                "average_fps": weighted_avg_fps,
            }
        )

    summary = {
        "runs_directory": str(runs_dir),
        "run_files_count": len(run_files),
        "algorithms": aggregate_rows,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[FPS] Aggregate Summary")
    for row in aggregate_rows:
        print(
            f"  - {row['algorithm']}: avg_fps={row['average_fps']:.2f} "
            f"(runs={row['runs']}, frames={row['total_frames']}, duration={row['total_duration_seconds']:.1f}s)"
        )
    print(f"[FPS] JSON: {out_path}")


if __name__ == "__main__":
    main()
