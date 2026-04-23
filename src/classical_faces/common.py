import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np

from src.dataset_layout import gather_augmented_person_dirs, infer_target_split_name

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMG_SIZE = (100, 100)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def root_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def resolve_path(path_value: str) -> str:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)
    return str(PROJECT_ROOT.joinpath(candidate))


def is_image_file(file_name: str) -> bool:
    _, ext = os.path.splitext(file_name)
    return ext.lower() in ALLOWED_EXTENSIONS


def get_person_dirs_from_processed(processed_root: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(processed_root):
        return []

    out: List[Tuple[str, str]] = []
    for person in sorted(os.listdir(processed_root)):
        person_path = os.path.join(processed_root, person)
        if os.path.isdir(person_path):
            out.append((person, person_path))
    return out


def get_person_dirs_from_raw(raw_root: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(raw_root):
        return []

    out: List[Tuple[str, str]] = []
    for person in sorted(os.listdir(raw_root)):
        person_path = os.path.join(raw_root, person)
        if os.path.isdir(person_path):
            out.append((person, person_path))
    return out


def get_person_dirs_from_augmented(
    augmented_root: str,
    include_splits: set[str],
    target_split: str | None = None,
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for _, person, person_path in gather_augmented_person_dirs(
        augmented_root=augmented_root,
        aug_splits=include_splits,
        target_split=target_split,
    ):
        out.append((person, person_path))
    return out


def gather_dataset_entries(
    base_data_dir: str,
    raw_dir: str,
    processed_dir: str,
    aug_dir: str,
    aug_splits: set[str],
    include_processed: bool,
    include_augmented: bool,
) -> list[tuple[str, str, str]]:
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
        target_split = infer_target_split_name(raw_dir=raw_dir, processed_dir=processed_dir)
        entries.extend(
            gather_augmented_person_dirs(
                augmented_root=augmented_root,
                aug_splits=aug_splits,
                target_split=target_split,
            )
        )

    return entries


def detect_face_or_fallback(
    image_gray: np.ndarray,
    face_cascade: cv.CascadeClassifier,
    min_face_size: int,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> tuple[np.ndarray | None, bool | None]:
    detected = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size),
    )

    if len(detected) > 0:
        x, y, w, h = max(detected, key=lambda box: box[2] * box[3])
        roi = image_gray[y : y + h, x : x + w]
        used_fallback = False
    else:
        if image_gray.shape[0] < min_face_size or image_gray.shape[1] < min_face_size:
            return None, None
        roi = image_gray
        used_fallback = True

    roi = cv.equalizeHist(roi)
    roi = cv.resize(roi, IMG_SIZE)
    return roi, used_fallback


def preprocess_direct_full_image(image_gray: np.ndarray, min_face_size: int) -> np.ndarray | None:
    if image_gray.shape[0] < min_face_size or image_gray.shape[1] < min_face_size:
        return None
    roi = cv.equalizeHist(image_gray)
    return cv.resize(roi, IMG_SIZE)


def maybe_downscale(gray: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return gray
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return gray
    scale = max_side / float(longest)
    return cv.resize(gray, (int(w * scale), int(h * scale)))


@dataclass
class Stats:
    total_images: int = 0
    evaluated_images: int = 0
    correct: int = 0
    predicted_known: int = 0
    predicted_unknown: int = 0
    face_detected: int = 0
    face_fallback_used: int = 0
    skipped_unreadable: int = 0
    skipped_too_small: int = 0
    skipped_unseen_identity: int = 0


def summarize_bucket(name: str, stats: Stats) -> str:
    eval_count = stats.evaluated_images
    hit_rate = (100.0 * stats.correct / eval_count) if eval_count else 0.0
    known_rate = (100.0 * stats.predicted_known / eval_count) if eval_count else 0.0
    unknown_rate = (100.0 * stats.predicted_unknown / eval_count) if eval_count else 0.0

    return (
        f"{name:<18} total={stats.total_images:<6} eval={eval_count:<6} "
        f"hit={hit_rate:6.2f}% known={known_rate:6.2f}% unknown={unknown_rate:6.2f}% "
        f"detected={stats.face_detected:<6} fallback={stats.face_fallback_used:<6}"
    )


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
        "known_rate_percent": (100.0 * stats.predicted_known / eval_count) if eval_count else 0.0,
        "unknown_rate_percent": (100.0 * stats.predicted_unknown / eval_count) if eval_count else 0.0,
        "face_detected": stats.face_detected,
        "face_fallback_used": stats.face_fallback_used,
        "skipped_unreadable": stats.skipped_unreadable,
        "skipped_too_small": stats.skipped_too_small,
        "skipped_unseen_identity": stats.skipped_unseen_identity,
    }


def compute_threshold_sweep(eval_records: list[dict], thresholds: list[float]) -> list[dict]:
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
                best_name = r["best_name"]
                dist = float(r["distance"])
                pred = best_name if (best_name != "Unknown" and dist <= thr) else "Unknown"
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


def format_seconds(seconds: float) -> str:
    secs = max(0, int(seconds))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_progress(
    processed: int,
    total: int,
    start_time: float,
    status: str,
) -> None:
    if total <= 0:
        return

    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - processed)
    eta = remaining / rate if rate > 0 else 0.0
    pct = (100.0 * processed / total)

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
