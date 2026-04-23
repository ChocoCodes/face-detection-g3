import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np

from src.dataset_layout import gather_augmented_person_dirs, infer_target_split_name

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMG_SIZE = (100, 100)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WORKER_CASCADE: cv.CascadeClassifier | None = None
_WORKER_MIN_FACE_SIZE = 40


def root_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def resolve_path(path_value: str) -> str:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)
    return str(PROJECT_ROOT.joinpath(candidate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Eigenfaces face recognizer from split/train by default."
    )
    parser.add_argument(
        "--base-data-dir",
        default=root_path("data"),
        help="Base data directory that contains augmented41mods and optional raw/processed folders.",
    )
    parser.add_argument(
        "--raw-dir-name",
        default="lasalle_db1",
        help="Folder name for training dataset inside base-data-dir.",
    )
    parser.add_argument(
        "--include-raw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include training dataset folder during training.",
    )
    parser.add_argument(
        "--processed-dir-name",
        default="lasalle_db1_processed",
        help="Folder name for processed dataset inside base-data-dir.",
    )
    parser.add_argument(
        "--augmented-dir-name",
        default="augmented41mods",
        help="Folder name for augmented dataset inside base-data-dir.",
    )
    parser.add_argument(
        "--aug-splits",
        default="heavy,medium,light",
        help="Comma-separated augmented subsets to include (e.g. heavy,medium,light).",
    )
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help="Include processed dataset during training.",
    )
    parser.add_argument(
        "--model-output",
        default=root_path("models", "eigenfaces", "trainer_eigenfaces.yml"),
        help="Path for the trained Eigenfaces model output.",
    )
    parser.add_argument(
        "--labels-output",
        default=root_path("models", "eigenfaces", "labels_eigenfaces.json"),
        help="Path for the labels JSON output.",
    )
    parser.add_argument(
        "--max-images-per-person",
        type=int,
        default=0,
        help="Optional cap per person (0 = no cap).",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=40,
        help="Minimum detected face width/height.",
    )
    parser.add_argument(
        "--cascade-path",
        default=root_path("haar", "haarcascade_frontalface_default.xml"),
        help="Path to Haar cascade XML.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Progress update interval in images.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Image preprocessing worker processes (1 = single-process).",
    )
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=0,
        help="OpenCV internal thread count (0 = OpenCV default).",
    )
    return parser.parse_args()


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


def detect_face_or_fallback(
    image_gray: np.ndarray,
    face_cascade: cv.CascadeClassifier,
    min_face_size: int,
) -> tuple[np.ndarray | None, bool | None]:
    detected = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=1.1,
        minNeighbors=5,
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
    return cv.resize(roi, IMG_SIZE), used_fallback


def _init_pool_worker(cascade_path: str, min_face_size: int) -> None:
    global _WORKER_CASCADE, _WORKER_MIN_FACE_SIZE
    _WORKER_CASCADE = cv.CascadeClassifier(cascade_path)
    _WORKER_MIN_FACE_SIZE = min_face_size
    if _WORKER_CASCADE.empty():
        raise FileNotFoundError(f"Could not load cascade file in worker: {cascade_path}")


def _process_image_in_worker(task: Tuple[str, int, str]) -> Tuple[np.ndarray | None, int, str, bool | None]:
    image_path, label_id, person = task
    img = cv.imread(image_path)
    if img is None:
        return None, label_id, person, None

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if _WORKER_CASCADE is None:
        return None, label_id, person, None
    face, used_fallback = detect_face_or_fallback(gray, _WORKER_CASCADE, _WORKER_MIN_FACE_SIZE)
    return face, label_id, person, used_fallback


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


def count_planned_images(person_dirs: List[Tuple[str, str]], max_images_per_person: int) -> int:
    if max_images_per_person <= 0:
        total = 0
        for _, folder in person_dirs:
            total += len([f for f in sorted(os.listdir(folder)) if is_image_file(f)])
        return total

    person_counts: Dict[str, int] = defaultdict(int)
    total = 0
    for person, folder in person_dirs:
        image_files = [f for f in sorted(os.listdir(folder)) if is_image_file(f)]
        remaining = max_images_per_person - person_counts[person]
        if remaining <= 0:
            continue
        used = min(len(image_files), remaining)
        person_counts[person] += used
        total += used
    return total


def gather_training_data(
    person_dirs: List[Tuple[str, str]],
    face_cascade: cv.CascadeClassifier,
    cascade_path: str,
    max_images_per_person: int,
    min_face_size: int,
    progress_interval: int,
    workers: int,
) -> Tuple[List[np.ndarray], List[int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    faces: List[np.ndarray] = []
    labels: List[int] = []
    label_map: Dict[str, int] = {}
    person_counts: Dict[str, int] = defaultdict(int)

    total_planned = count_planned_images(person_dirs, max_images_per_person)
    processed_images = 0
    start_time = time.time()

    stats = {
        "skipped_images": 0,
        "face_detected": 0,
        "face_fallback_used": 0,
    }

    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_pool_worker,
            initargs=(cascade_path, min_face_size),
        )

    try:
        for person, folder in person_dirs:
            if person not in label_map:
                label_map[person] = len(label_map)

            label_id = label_map[person]
            image_files = [f for f in sorted(os.listdir(folder)) if is_image_file(f)]

            if max_images_per_person > 0:
                remaining = max_images_per_person - person_counts[person]
                if remaining <= 0:
                    continue
                image_files = image_files[:remaining]

            if pool is None:
                for image_name in image_files:
                    processed_images += 1
                    if (
                        processed_images == 1
                        or processed_images % max(1, progress_interval) == 0
                        or processed_images == total_planned
                    ):
                        print_progress(
                            processed=processed_images,
                            total=total_planned,
                            start_time=start_time,
                            status=f"load/{person}",
                        )

                    image_path = os.path.join(folder, image_name)
                    img = cv.imread(image_path)
                    if img is None:
                        stats["skipped_images"] += 1
                        continue

                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    face, used_fallback = detect_face_or_fallback(gray, face_cascade, min_face_size)
                    if face is None:
                        stats["skipped_images"] += 1
                        continue

                    if used_fallback:
                        stats["face_fallback_used"] += 1
                    else:
                        stats["face_detected"] += 1

                    faces.append(face)
                    labels.append(label_id)
                    person_counts[person] += 1
                continue

            tasks: List[Tuple[str, int, str]] = []
            for image_name in image_files:
                tasks.append((os.path.join(folder, image_name), label_id, person))

            for face, out_label_id, out_person, used_fallback in pool.map(
                _process_image_in_worker,
                tasks,
                chunksize=16,
            ):
                processed_images += 1
                if (
                    processed_images == 1
                    or processed_images % max(1, progress_interval) == 0
                    or processed_images == total_planned
                ):
                    print_progress(
                        processed=processed_images,
                        total=total_planned,
                        start_time=start_time,
                        status=f"load/{out_person}",
                    )

                if face is None:
                    stats["skipped_images"] += 1
                    continue

                if used_fallback:
                    stats["face_fallback_used"] += 1
                else:
                    stats["face_detected"] += 1

                faces.append(face)
                labels.append(out_label_id)
                person_counts[out_person] += 1
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if total_planned > 0:
        print_progress(processed_images, total_planned, start_time, "done")
        print()

    return faces, labels, label_map, dict(person_counts), stats


def main() -> None:
    args = parse_args()
    args.workers = max(1, int(args.workers))

    args.base_data_dir = resolve_path(args.base_data_dir)
    args.model_output = resolve_path(args.model_output)
    args.labels_output = resolve_path(args.labels_output)
    args.cascade_path = resolve_path(args.cascade_path)

    raw_root = os.path.join(args.base_data_dir, args.raw_dir_name)
    processed_root = os.path.join(args.base_data_dir, args.processed_dir_name)
    augmented_root = os.path.join(args.base_data_dir, args.augmented_dir_name)
    include_splits = {
        split.strip().lower()
        for split in args.aug_splits.split(",")
        if split.strip()
    }

    print("[INFO] Initializing Haar cascade and Eigenfaces recognizer...")
    face_cascade = cv.CascadeClassifier(args.cascade_path)
    if face_cascade.empty():
        raise FileNotFoundError(f"Could not load cascade file: {args.cascade_path}")
    if args.opencv_threads > 0:
        cv.setNumThreads(args.opencv_threads)
        print(f"[INFO] OpenCV threads set to: {args.opencv_threads}")

    recognizer = cv.face.EigenFaceRecognizer_create()

    print(f"[INFO] Using training dataset: {raw_root} (enabled={args.include_raw})")
    print(f"[INFO] Using processed dataset: {processed_root} (enabled={args.include_processed})")
    print(f"[INFO] Using augmented dataset: {augmented_root}")
    print(f"[INFO] Included augmented splits: {sorted(include_splits)}")
    print(
        f"[INFO] Image preprocessing mode: "
        f"{'multiprocess' if args.workers > 1 else 'single-process'} (workers={args.workers})"
    )

    raw_people = get_person_dirs_from_raw(raw_root) if args.include_raw else []
    processed_people = get_person_dirs_from_processed(processed_root) if args.include_processed else []
    target_split = infer_target_split_name(
        raw_dir=args.raw_dir_name,
        processed_dir=args.processed_dir_name,
    )
    augmented_people = get_person_dirs_from_augmented(
        augmented_root,
        include_splits,
        target_split=target_split,
    )
    person_dirs = raw_people + processed_people + augmented_people

    if not person_dirs:
        raise RuntimeError(
            "No training folders found. Check base-data-dir and dataset folder names."
        )

    total_start = time.time()
    faces, labels, label_map, person_counts, stats = gather_training_data(
        person_dirs=person_dirs,
        face_cascade=face_cascade,
        cascade_path=args.cascade_path,
        max_images_per_person=args.max_images_per_person,
        min_face_size=args.min_face_size,
        progress_interval=args.progress_interval,
        workers=args.workers,
    )

    if not faces:
        raise RuntimeError("No training faces extracted. Cannot train model.")

    print(f"[INFO] Training samples collected: {len(faces)}")
    print(f"[INFO] Identities: {len(label_map)}")
    print(f"[INFO] Skipped images: {stats['skipped_images']}")
    print(f"[INFO] Face detected: {stats['face_detected']}")
    print(f"[INFO] Face fallback used: {stats['face_fallback_used']}")

    print("[INFO] Starting Eigenfaces training (this stage has no internal progress output)...")
    train_start = time.time()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    train_time = time.time() - train_start

    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.labels_output), exist_ok=True)
    recognizer.save(args.model_output)
    with open(args.labels_output, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    total_time = time.time() - total_start

    print(f"[OK] Saved model to: {args.model_output}")
    print(f"[OK] Saved labels to: {args.labels_output}")
    print(f"[TIME] Train time: {train_time:.2f}s")
    print(f"[TIME] Total time: {total_time:.2f}s")

    print("[INFO] Per-person sample counts:")
    for person in sorted(person_counts):
        print(f"  - {person}: {person_counts[person]}")


if __name__ == "__main__":
    main()
