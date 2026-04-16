#!/usr/bin/env python3
"""
Evaluate ArcFace enrollment on La Salle test datasets.

Provides bucketed metrics, threshold sweep, and detailed reporting for
robust real-world performance assessment across diverse conditions.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np

try:
    import insightface
except ImportError:
    print("[ERROR] insightface not installed. Run: pip install insightface")
    sys.exit(1)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
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
        description="Evaluate ArcFace enrollment on La Salle datasets."
    )
    parser.add_argument("--base-data-dir", default=root_path("data", "split"))
    parser.add_argument("--raw-dir-name", default="test")
    parser.add_argument("--processed-dir-name", default="lasalle_db1_processed")
    parser.add_argument("--augmented-dir-name", default="augmented41mods")
    parser.add_argument(
        "--include-processed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include processed dataset in evaluation.",
    )
    parser.add_argument(
        "--include-augmented",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include augmented dataset in evaluation.",
    )
    parser.add_argument("--aug-splits", default="original,light,medium,heavy")
    parser.add_argument(
        "--model-dir",
        default=root_path("models", "arcface_mobilenet"),
        help="Directory containing ArcFace model.",
    )
    parser.add_argument(
        "--enrollment-path",
        default=root_path("models", "arcface_mobilenet", "enrollment.json"),
        help="Path to enrollment JSON.",
    )
    parser.add_argument(
        "--report-json",
        default=root_path("reports", "evaluation", "arcface_eval.json"),
        help="Output report JSON.",
    )
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--max-images-per-person", type=int, default=0)
    parser.add_argument(
        "--threshold-sweep",
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        help="Thresholds for sweep analysis.",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-chunksize", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--show-misclassified", type=int, default=20)
    parser.add_argument(
        "--exclude-identities",
        default="",
        help="Comma-separated identity names to exclude from evaluation.",
    )
    return parser.parse_args()


def is_image_file(file_name: str) -> bool:
    _, ext = os.path.splitext(file_name)
    return ext.lower() in ALLOWED_EXTENSIONS


@dataclass
class Stats:
    total_images: int = 0
    evaluated_images: int = 0
    correct: int = 0
    predicted_known: int = 0
    predicted_unknown: int = 0
    face_detected: int = 0
    skipped_unreadable: int = 0
    skipped_no_face: int = 0
    skipped_unseen_identity: int = 0


@dataclass
class EvalSample:
    bucket: str
    person: str
    path: str


_EVAL_AFACE = None
_EVAL_CENTROIDS = {}
_EVAL_PEOPLE = []
_EVAL_THRESHOLD = 0.55


def _init_eval_worker(
    model_dir: str,
    centroids_payload: dict,
    threshold: float,
) -> None:
    global _EVAL_AFACE, _EVAL_CENTROIDS, _EVAL_PEOPLE, _EVAL_THRESHOLD
    
    try:
        _EVAL_AFACE = insightface.app.FaceAnalysis(
            name="buffalo_s",
            root=model_dir,
            providers=["CPUExecutionProvider"],
        )
        _EVAL_AFACE.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
    except Exception as e:
        print(f"[ERROR] Failed to load ArcFace: {e}")
        raise
    
    _EVAL_CENTROIDS = {
        person: np.array(vec, dtype=np.float32)
        for person, vec in centroids_payload.items()
    }
    _EVAL_PEOPLE = sorted(_EVAL_CENTROIDS.keys())
    _EVAL_THRESHOLD = threshold


def _process_eval_sample(sample: EvalSample) -> dict:
    """Evaluate single sample."""
    result = {
        "bucket": sample.bucket,
        "truth": sample.person,
        "status": "",
        "predicted": "Unknown",
        "score": -1.0,
        "path": sample.path,
    }
    
    img = cv.imread(sample.path)
    if img is None:
        result["status"] = "unreadable"
        return result
    
    if sample.person not in _EVAL_CENTROIDS:
        result["status"] = "unseen_identity"
        return result
    
    try:
        faces = _EVAL_AFACE.get(img)
        if not faces:
            result["status"] = "no_face"
            return result
        
        best_face = max(faces, key=lambda f: f.det_score)
        emb = best_face.embedding.astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-12:
            emb = emb / norm
        
        # Match against enrollment
        best_person = "Unknown"
        best_score = -1.0
        for person in _EVAL_PEOPLE:
            score = float(np.dot(emb, _EVAL_CENTROIDS[person]))
            if score > best_score:
                best_score = score
                best_person = person
        
        predicted = best_person if best_score >= _EVAL_THRESHOLD else "Unknown"
        result["status"] = "evaluated"
        result["best_person"] = best_person
        result["predicted"] = predicted
        result["score"] = float(best_score)
        return result
    except Exception as e:
        result["status"] = "error"
        return result


def _process_eval_batch(batch_samples: List[EvalSample]) -> List[dict]:
    return [_process_eval_sample(sample) for sample in batch_samples]


def make_batches(samples: List[EvalSample], batch_size: int) -> List[List[EvalSample]]:
    size = max(1, batch_size)
    return [samples[i : i + size] for i in range(0, len(samples), size)]


def gather_entries(
    base_data_dir: str,
    raw_dir: str,
    processed_dir: str,
    aug_dir: str,
    aug_splits: set[str],
    include_processed: bool,
    include_augmented: bool,
):
    entries = []
    
    raw_root = os.path.join(base_data_dir, raw_dir)
    if os.path.isdir(raw_root):
        for person in sorted(os.listdir(raw_root)):
            person_path = os.path.join(raw_root, person)
            if os.path.isdir(person_path):
                entries.append(("raw", person, person_path))
    
    if include_processed:
        processed_root = os.path.join(base_data_dir, processed_dir)
        if os.path.isdir(processed_root):
            for person in sorted(os.listdir(processed_root)):
                person_path = os.path.join(processed_root, person)
                if os.path.isdir(person_path):
                    entries.append(("processed", person, person_path))
    
    if include_augmented:
        augmented_root = os.path.join(base_data_dir, aug_dir)
        if os.path.isdir(augmented_root):
            for split_name in sorted(os.listdir(augmented_root)):
                if aug_splits and split_name.lower() not in aug_splits:
                    continue
                split_path = os.path.join(augmented_root, split_name)
                if not os.path.isdir(split_path):
                    continue
                bucket = f"augmented/{split_name}"
                for person in sorted(os.listdir(split_path)):
                    person_path = os.path.join(split_path, person)
                    if os.path.isdir(person_path):
                        entries.append((bucket, person, person_path))
    
    return entries


def build_eval_samples(entries, max_images_per_person: int) -> List[EvalSample]:
    samples: List[EvalSample] = []
    for bucket_name, person_name, person_path in entries:
        files = [f for f in sorted(os.listdir(person_path)) if is_image_file(f)]
        if max_images_per_person > 0:
            files = files[:max_images_per_person]
        for image_name in files:
            samples.append(
                EvalSample(
                    bucket=bucket_name,
                    person=person_name,
                    path=os.path.join(person_path, image_name),
                )
            )
    return samples


def format_seconds(seconds: float) -> str:
    secs = max(0, int(seconds))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_progress(processed: int, total: int, start_time: float, status: str) -> None:
    if total <= 0:
        return
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - processed)
    eta = remaining / rate if rate > 0 else 0.0
    pct = 100.0 * processed / total
    bar_width = 28
    filled = int(bar_width * processed / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    msg = (
        f"\r[PROGRESS] [{bar}] {processed}/{total} ({pct:6.2f}%) "
        f"| elapsed {format_seconds(elapsed)} | eta {format_seconds(eta)} "
        f"| {rate:6.2f} img/s | {status:<28}"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()


def bucket_to_dict(name: str, stats: Stats) -> dict:
    eval_count = stats.evaluated_images
    return {
        "bucket": name,
        "total_images": stats.total_images,
        "evaluated_images": eval_count,
        "correct": stats.correct,
        "hit_rate_percent": (100.0 * stats.correct / eval_count) if eval_count else 0.0,
        "predicted_known": stats.predicted_known,
        "predicted_unknown": stats.predicted_unknown,
        "face_detected": stats.face_detected,
        "skipped_unreadable": stats.skipped_unreadable,
        "skipped_no_face": stats.skipped_no_face,
        "skipped_unseen_identity": stats.skipped_unseen_identity,
    }


def compute_threshold_sweep(eval_records: List[dict], thresholds: List[float]) -> List[dict]:
    sweep = []
    buckets = sorted({r["bucket"] for r in eval_records})
    
    for thr in thresholds:
        overall_total = len(eval_records)
        overall_correct = 0
        per_bucket = {}
        
        for bucket in buckets:
            bucket_records = [r for r in eval_records if r["bucket"] == bucket]
            total = len(bucket_records)
            correct = 0
            for r in bucket_records:
                pred = r["best_person"] if r["score"] >= thr else "Unknown"
                if pred == r["truth"]:
                    correct += 1
            per_bucket[bucket] = {
                "total": total,
                "correct": correct,
                "hit_rate_percent": (100.0 * correct / total) if total else 0.0,
            }
            overall_correct += correct
        
        sweep.append(
            {
                "threshold": thr,
                "overall_total": overall_total,
                "overall_correct": overall_correct,
                "overall_hit_rate_percent": (100.0 * overall_correct / overall_total)
                if overall_total
                else 0.0,
                "by_bucket": per_bucket,
            }
        )
    
    return sweep


def main() -> None:
    args = parse_args()
    args.base_data_dir = resolve_path(args.base_data_dir)
    args.model_dir = resolve_path(args.model_dir)
    args.enrollment_path = resolve_path(args.enrollment_path)
    args.report_json = resolve_path(args.report_json)
    
    if not os.path.exists(args.enrollment_path):
        print(f"[ERROR] Enrollment not found: {args.enrollment_path}")
        print(f"[HINT] Run: python src/arcface_mobilenet/trainer.py")
        sys.exit(1)
    
    with open(args.enrollment_path, "r", encoding="utf-8") as f:
        enrollment = json.load(f)
    
    people = sorted(enrollment["people"].keys())
    centroids_payload = {
        person: enrollment["people"][person]["centroid"]
        for person in people
    }
    
    if args.threshold < 0:
        args.threshold = float(enrollment["metadata"].get("recommended_threshold", 0.55))
    
    workers = max(1, args.workers)
    worker_chunksize = max(1, args.worker_chunksize)
    batch_size = max(1, args.batch_size)
    progress_interval = max(1, args.progress_interval)
    
    aug_splits = {s.strip().lower() for s in args.aug_splits.split(",") if s.strip()}
    excluded_identities = {
        s.strip().lower() for s in args.exclude_identities.split(",") if s.strip()
    }
    entries = gather_entries(
        base_data_dir=args.base_data_dir,
        raw_dir=args.raw_dir_name,
        processed_dir=args.processed_dir_name,
        aug_dir=args.augmented_dir_name,
        aug_splits=aug_splits,
        include_processed=args.include_processed,
        include_augmented=args.include_augmented,
    )

    if excluded_identities:
        entries = [e for e in entries if e[1].lower() not in excluded_identities]
    
    if not entries:
        print("[ERROR] No dataset found.")
        sys.exit(1)
    
    samples = build_eval_samples(entries, args.max_images_per_person)
    total_planned = len(samples)
    
    per_bucket = defaultdict(Stats)
    overall = Stats()
    misclassified = []
    eval_records: List[dict] = []
    
    def aggregate_result(result: dict) -> None:
        bucket_name = result["bucket"]
        per_bucket[bucket_name].total_images += 1
        overall.total_images += 1
        
        status = result["status"]
        if status == "unreadable":
            per_bucket[bucket_name].skipped_unreadable += 1
            overall.skipped_unreadable += 1
            return
        if status == "unseen_identity":
            per_bucket[bucket_name].skipped_unseen_identity += 1
            overall.skipped_unseen_identity += 1
            return
        if status == "no_face":
            per_bucket[bucket_name].skipped_no_face += 1
            overall.skipped_no_face += 1
            return
        if status != "evaluated":
            return
        
        per_bucket[bucket_name].evaluated_images += 1
        overall.evaluated_images += 1
        per_bucket[bucket_name].face_detected += 1
        overall.face_detected += 1
        
        truth = result["truth"]
        predicted = result["predicted"]
        
        if predicted == "Unknown":
            per_bucket[bucket_name].predicted_unknown += 1
            overall.predicted_unknown += 1
        else:
            per_bucket[bucket_name].predicted_known += 1
            overall.predicted_known += 1
        
        if predicted == truth:
            per_bucket[bucket_name].correct += 1
            overall.correct += 1
        elif len(misclassified) < args.show_misclassified:
            misclassified.append(
                {
                    "bucket": bucket_name,
                    "truth": truth,
                    "predicted": predicted,
                    "score": float(result["score"]),
                    "path": result["path"],
                }
            )
        
        eval_records.append(
            {
                "bucket": bucket_name,
                "truth": truth,
                "best_person": result.get("best_person", "Unknown"),
                "score": float(result["score"]),
            }
        )
    
    start = time.time()
    
    if workers == 1:
        try:
            aface = insightface.app.FaceAnalysis(
                name="buffalo_s",
                root=args.model_dir,
                providers=["CPUExecutionProvider"],
            )
            aface.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
        except Exception as e:
            print(f"[ERROR] Failed to load ArcFace: {e}")
            sys.exit(1)
        
        _init_eval_worker(args.model_dir, centroids_payload, args.threshold)
        
        for idx, sample in enumerate(samples, start=1):
            result = _process_eval_sample(sample)
            aggregate_result(result)
            if idx == 1 or idx % progress_interval == 0 or idx == total_planned:
                print_progress(idx, total_planned, start, f"{sample.bucket}/{sample.person}")
    else:
        batches = make_batches(samples, batch_size)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_eval_worker,
            initargs=(args.model_dir, centroids_payload, args.threshold),
        ) as executor:
            idx = 0
            for batch_results in executor.map(
                _process_eval_batch,
                batches,
                chunksize=worker_chunksize,
            ):
                for result in batch_results:
                    idx += 1
                    aggregate_result(result)
                    if idx == 1 or idx % progress_interval == 0 or idx == total_planned:
                        print_progress(idx, total_planned, start, f"{result['bucket']}/{result['truth']}")
    
    elapsed = time.time() - start
    if total_planned > 0:
        print_progress(min(len(eval_records), total_planned), total_planned, start, "done")
        print()
    
    thresholds = [float(x.strip()) for x in args.threshold_sweep.split(",") if x.strip()]
    threshold_sweep = compute_threshold_sweep(eval_records, thresholds) if thresholds else []
    
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    
    report = {
        "model_family": "arcface_buffalo_s",
        "enrollment_path": args.enrollment_path,
        "threshold": args.threshold,
        "config": {
            "progress_interval": args.progress_interval,
            "workers": workers,
            "batch_size": batch_size,
            "threshold_sweep": args.threshold_sweep,
            "exclude_identities": sorted(excluded_identities),
        },
        "elapsed_seconds": elapsed,
        "buckets": [bucket_to_dict(name, per_bucket[name]) for name in sorted(per_bucket.keys())],
        "overall": bucket_to_dict("overall", overall),
        "threshold_sweep": threshold_sweep,
        "sample_misclassifications": misclassified,
    }
    
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print("\n[RESULT] ArcFace Evaluation")
    print(f"[INFO] Model: ArcFace Buffalo-S (Real-world Robust)")
    print(f"[INFO] Threshold: {args.threshold:.3f}")
    print(f"[INFO] Evaluated in: {elapsed:.2f}s")
    print("\n[HIT RATE BY BUCKET]")
    for name in sorted(per_bucket.keys()):
        stats = per_bucket[name]
        hit = (100.0 * stats.correct / stats.evaluated_images) if stats.evaluated_images else 0.0
        print(f"{name:20} total={stats.total_images:4d} eval={stats.evaluated_images:4d} hit={hit:6.2f}%")
    print("\n[OVERALL]")
    overall_hit = (100.0 * overall.correct / overall.evaluated_images) if overall.evaluated_images else 0.0
    print(f"Overall hit rate: {overall_hit:.2f}%")
    print(f"\n[OK] Report: {args.report_json}")


if __name__ == "__main__":
    main()
