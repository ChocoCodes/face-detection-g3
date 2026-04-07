#!/usr/bin/env python3
"""
Train ArcFace enrollment on La Salle datasets with robust real-world handling.

Uses InsightFace ArcFace models (buffalo_s, buffalo_l, or similar) for:
- Better accuracy than MobileFaceNet on diverse conditions
- Built-in face alignment and preprocessing
- Proven robustness across lighting, distance, pose variations
- Fixed roster optimal performance

Features:
- Multiprocessing support for fast enrollment building
- Per-identity weighting (raw samples weighted higher)
- Threshold calibration
- Support for raw/processed/augmented data
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

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
        description="Build ArcFace enrollment for robust real-world face recognition."
    )
    parser.add_argument("--base-data-dir", default=root_path("data"))
    parser.add_argument("--raw-dir-name", default="lasalle_db1")
    parser.add_argument("--processed-dir-name", default="lasalle_db1_processed")
    parser.add_argument("--augmented-dir-name", default="augmented41mods")
    parser.add_argument(
        "--aug-splits",
        default="original,light,medium",
        help="Comma-separated augmented subsets to include.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw dataset (recommended for diversity).",
    )
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help="Include processed dataset during enrollment build.",
    )
    parser.add_argument(
        "--max-images-per-person",
        type=int,
        default=0,
        help="Cap per person per bucket (0 = no cap).",
    )
    parser.add_argument(
        "--model-dir",
        default=root_path("models", "arcface_mobilenet"),
        help="Directory containing ArcFace model files.",
    )
    parser.add_argument(
        "--enrollment-output",
        default=root_path("models", "arcface_mobilenet", "enrollment.json"),
        help="Output enrollment JSON.",
    )
    parser.add_argument(
        "--raw-weight",
        type=float,
        default=1.5,
        help="Weight multiplier for raw samples in centroid computation.",
    )
    parser.add_argument(
        "--threshold-candidates",
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        help="Cosine thresholds to evaluate.",
    )
    parser.add_argument(
        "--calibration-objective",
        choices=["balanced_accuracy", "id_accuracy"],
        default="balanced_accuracy",
        help="Objective for threshold selection.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--worker-chunksize", type=int, default=64)
    parser.add_argument("--progress-interval", type=int, default=50)
    return parser.parse_args()


def is_image_file(file_name: str) -> bool:
    _, ext = os.path.splitext(file_name)
    return ext.lower() in ALLOWED_EXTENSIONS


@dataclass
class Sample:
    bucket: str
    person: str
    path: str


_WORKER_AFACE = None
_WORKER_ASSUME_PROCESSED = False
_WORKER_RAW_WEIGHT = 1.5


def _init_worker(model_dir: str, assume_processed_are_cropped: bool, raw_weight: float) -> None:
    global _WORKER_AFACE, _WORKER_ASSUME_PROCESSED, _WORKER_RAW_WEIGHT
    
    try:
        _WORKER_AFACE = insightface.app.FaceAnalysis(
            name="buffalo_s",
            root=model_dir,
            providers=["CPUExecutionProvider"],
        )
        _WORKER_AFACE.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
    except Exception as e:
        print(f"[ERROR] Failed to load ArcFace model: {e}")
        raise
    
    _WORKER_ASSUME_PROCESSED = assume_processed_are_cropped
    _WORKER_RAW_WEIGHT = raw_weight


def _extract_embedding(
    sample: Sample,
    aface,
    assume_processed_are_cropped: bool,
    raw_weight: float,
) -> tuple[str, np.ndarray | None]:
    """Extract ArcFace embedding from image."""
    img = cv.imread(sample.path)
    if img is None:
        return "unreadable", None
    
    try:
        # ArcFace internally handles detection, alignment, preprocessing
        faces = aface.get(img)
        if not faces:
            return "no_face", None
        
        # Use the best (most confident) face
        best_face = max(faces, key=lambda f: f.det_score)
        embedding = best_face.embedding.astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-12:
            embedding = embedding / norm
        
        return "ok", embedding
    except Exception as e:
        print(f"[ERROR] Extraction failed for {sample.path}: {e}")
        return "error", None


def _process_sample_worker(sample: Sample) -> tuple[str, str, str, list | None]:
    status, emb = _extract_embedding(
        sample=sample,
        aface=_WORKER_AFACE,
        assume_processed_are_cropped=_WORKER_ASSUME_PROCESSED,
        raw_weight=_WORKER_RAW_WEIGHT,
    )
    if emb is None:
        return status, sample.person, sample.bucket, None
    return status, sample.person, sample.bucket, emb.tolist()


def _process_batch_worker(batch_samples: List[Sample]) -> List[tuple]:
    out = []
    for sample in batch_samples:
        result = _process_sample_worker(sample)
        out.append(result)
    return out


def make_batches(samples: List[Sample], batch_size: int) -> List[List[Sample]]:
    size = max(1, batch_size)
    return [samples[i : i + size] for i in range(0, len(samples), size)]


def gather_samples(
    base_data_dir: str,
    raw_dir: str,
    processed_dir: str,
    augmented_dir: str,
    aug_splits: set[str],
    include_raw: bool,
    include_processed: bool,
    max_images_per_person: int,
) -> List[Sample]:
    """Gather training samples from all data buckets."""
    samples: List[Sample] = []
    used = defaultdict(int)
    
    def add_bucket(bucket_name: str, person_name: str, person_path: str):
        files = [f for f in sorted(os.listdir(person_path)) if is_image_file(f)]
        for image_name in files:
            key = (bucket_name, person_name)
            if max_images_per_person > 0 and used[key] >= max_images_per_person:
                break
            used[key] += 1
            samples.append(
                Sample(
                    bucket=bucket_name,
                    person=person_name,
                    path=os.path.join(person_path, image_name),
                )
            )
    
    if include_raw:
        raw_root = os.path.join(base_data_dir, raw_dir)
        if os.path.isdir(raw_root):
            for person in sorted(os.listdir(raw_root)):
                person_path = os.path.join(raw_root, person)
                if os.path.isdir(person_path):
                    add_bucket("raw", person, person_path)
    
    if include_processed:
        processed_root = os.path.join(base_data_dir, processed_dir)
        if os.path.isdir(processed_root):
            for person in sorted(os.listdir(processed_root)):
                person_path = os.path.join(processed_root, person)
                if os.path.isdir(person_path):
                    add_bucket("processed", person, person_path)
    
    augmented_root = os.path.join(base_data_dir, augmented_dir)
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
                    add_bucket(bucket, person, person_path)
    
    return samples


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= 1e-12:
        return v
    return v / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def select_threshold_score(score_row: dict, objective: str) -> float:
    if objective == "id_accuracy":
        return float(score_row["id_accuracy"])
    return float(score_row["balanced_accuracy"])


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
        f"| {rate:6.2f} img/s | {status:<20}"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()


def main() -> None:
    args = parse_args()
    args.base_data_dir = resolve_path(args.base_data_dir)
    args.model_dir = resolve_path(args.model_dir)
    args.enrollment_output = resolve_path(args.enrollment_output)
    
    if not os.path.exists(args.model_dir):
        print(f"[ERROR] Model directory not found: {args.model_dir}")
        print(f"[HINT] Run: python src/arcface_mobilenet/setup_model.py")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(args.enrollment_output), exist_ok=True)
    
    workers = max(1, args.workers)
    progress_interval = max(1, args.progress_interval)
    batch_size = max(1, args.batch_size)
    worker_chunksize = max(1, args.worker_chunksize)
    
    # Load ArcFace model
    print("[INFO] Loading ArcFace model...")
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
    
    aug_splits = {s.strip().lower() for s in args.aug_splits.split(",") if s.strip()}
    samples = gather_samples(
        base_data_dir=args.base_data_dir,
        raw_dir=args.raw_dir_name,
        processed_dir=args.processed_dir_name,
        augmented_dir=args.augmented_dir_name,
        aug_splits=aug_splits,
        include_raw=args.include_raw,
        include_processed=args.include_processed,
        max_images_per_person=args.max_images_per_person,
    )
    
    if not samples:
        print("[ERROR] No training samples found.")
        sys.exit(1)
    
    print(f"[INFO] Total samples: {len(samples)}")
    
    per_person_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    per_person_weights: Dict[str, List[float]] = defaultdict(list)
    diagnostics = {
        "total_samples": len(samples),
        "used_samples": 0,
        "skipped": {"unreadable": 0, "no_face": 0, "error": 0},
        "bucket_counts": defaultdict(int),
    }
    
    start = time.time()
    status_name = "embedding extraction"
    
    if workers == 1:
        for idx, sample in enumerate(samples, start=1):
            status, emb = _extract_embedding(
                sample=sample,
                aface=aface,
                assume_processed_are_cropped=False,
                raw_weight=args.raw_weight,
            )
            
            if status == "ok":
                per_person_embeddings[sample.person].append(emb)
                sample_weight = args.raw_weight if sample.bucket == "raw" else 1.0
                per_person_weights[sample.person].append(sample_weight)
                diagnostics["used_samples"] += 1
                diagnostics["bucket_counts"][sample.bucket] += 1
            else:
                diagnostics["skipped"][status] += 1
            
            if idx % progress_interval == 0 or idx == len(samples):
                print_progress(idx, len(samples), start, status_name)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(args.model_dir, False, args.raw_weight),
        ) as executor:
            batches = make_batches(samples, batch_size)
            idx = 0
            for batch_results in executor.map(_process_batch_worker, batches, chunksize=worker_chunksize):
                for result in batch_results:
                    idx += 1
                    status, person, bucket, emb_list = result
                    
                    if status == "ok":
                        emb = np.array(emb_list, dtype=np.float32)
                        per_person_embeddings[person].append(emb)
                        sample_weight = args.raw_weight if bucket == "raw" else 1.0
                        per_person_weights[person].append(sample_weight)
                        diagnostics["used_samples"] += 1
                        diagnostics["bucket_counts"][bucket] += 1
                    else:
                        diagnostics["skipped"][status] += 1
                    
                    if idx % progress_interval == 0 or idx == len(samples):
                        print_progress(idx, len(samples), start, status_name)
    
    if len(samples) > 0:
        print_progress(len(samples), len(samples), start, "done")
        print()
    
    if not per_person_embeddings:
        print("[ERROR] No embeddings extracted.")
        sys.exit(1)
    
    # Build centroids
    centroids: Dict[str, np.ndarray] = {}
    sample_embeddings: List[tuple[str, np.ndarray]] = []
    for person, embs in per_person_embeddings.items():
        mat = np.vstack(embs)
        weights = np.array(per_person_weights[person], dtype=np.float32)
        weighted_sum = np.sum(mat * weights[:, None], axis=0)
        centroid = l2_normalize((weighted_sum / np.sum(weights)).astype(np.float32))
        centroids[person] = centroid
        for emb in embs:
            sample_embeddings.append((person, emb))
    
    # Threshold calibration
    candidate_thresholds = [
        float(x.strip()) for x in args.threshold_candidates.split(",") if x.strip()
    ]
    threshold_scores = []
    people = sorted(centroids.keys())
    
    for thr in candidate_thresholds:
        correct = 0
        known_total = 0
        known_above = 0
        unknown_total = 0
        unknown_below = 0
        
        for truth, emb in sample_embeddings:
            best_person = "Unknown"
            best_score = -1.0
            own_score = cosine(emb, centroids[truth])
            max_other = -1.0
            
            for person in people:
                score = cosine(emb, centroids[person])
                if score > best_score:
                    best_score = score
                    best_person = person
                if person != truth and score > max_other:
                    max_other = score
            
            pred = best_person if best_score >= thr else "Unknown"
            if pred == truth:
                correct += 1
            
            known_total += 1
            if own_score >= thr:
                known_above += 1
            if max_other >= -0.5:
                unknown_total += 1
                if max_other < thr:
                    unknown_below += 1
        
        acc = correct / len(sample_embeddings)
        tpr = (known_above / known_total) if known_total else 0.0
        tnr = (unknown_below / unknown_total) if unknown_total else 0.0
        threshold_scores.append(
            {
                "threshold": thr,
                "id_accuracy": acc,
                "known_accept_rate": tpr,
                "unknown_reject_rate": tnr,
                "balanced_accuracy": 0.5 * (tpr + tnr),
            }
        )
    
    best_threshold = max(
        threshold_scores,
        key=lambda x: select_threshold_score(x, args.calibration_objective),
    )["threshold"]
    
    # Save enrollment
    enrollment = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": "arcface_buffalo_s",
            "model_dir": args.model_dir,
            "embedding_dim": int(next(iter(centroids.values())).shape[0]),
            "recommended_threshold": float(best_threshold),
            "calibration_objective": args.calibration_objective,
            "threshold_scores": threshold_scores,
            "diagnostics": {
                "total_samples": diagnostics["total_samples"],
                "used_samples": diagnostics["used_samples"],
                "skipped": diagnostics["skipped"],
                "bucket_counts": dict(diagnostics["bucket_counts"]),
            },
            "training_config": {
                "include_raw": args.include_raw,
                "include_processed": args.include_processed,
                "aug_splits": sorted(aug_splits),
                "raw_weight": args.raw_weight,
                "max_images_per_person": args.max_images_per_person,
                "calibration_objective": args.calibration_objective,
                "workers": workers,
                "batch_size": batch_size,
            },
        },
        "people": {
            person: {
                "sample_count": len(per_person_embeddings[person]),
                "centroid": centroids[person].tolist(),
            }
            for person in sorted(centroids.keys())
        },
    }
    
    with open(args.enrollment_output, "w", encoding="utf-8") as f:
        json.dump(enrollment, f, indent=2)
    
    total_elapsed = time.time() - start
    print(f"[OK] Enrollment: {args.enrollment_output}")
    print(f"[INFO] People: {len(centroids)}")
    print(f"[INFO] Used samples: {diagnostics['used_samples']} / {diagnostics['total_samples']}")
    print(f"[INFO] Recommended threshold: {best_threshold:.3f}")
    print(f"[TIME] Total: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
