import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LBPH face recognizer from split/train by default."
    )
    parser.add_argument(
        "--base-data-dir",
        default=root_path("data", "split"),
        help="Base data directory that contains the train/eval split folders.",
    )
    parser.add_argument(
        "--raw-dir-name",
        default="train",
        help="Folder name for training dataset inside base-data-dir.",
    )
    parser.add_argument(
        "--include-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        default="original,light,medium",
        help="Comma-separated augmented subsets to include (e.g. original,light,medium).",
    )
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help="Include processed dataset during training.",
    )
    parser.add_argument(
        "--model-output",
        default=root_path("models", "lbph", "trainer_lasalle.yml"),
        help="Path for the trained LBPH model output.",
    )
    parser.add_argument(
        "--labels-output",
        default=root_path("models", "lbph", "labels_lasalle.json"),
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
) -> List[Tuple[str, str]]:
    if not os.path.isdir(augmented_root):
        return []

    out: List[Tuple[str, str]] = []
    for split_name in sorted(os.listdir(augmented_root)):
        if include_splits and split_name.lower() not in include_splits:
            continue

        split_path = os.path.join(augmented_root, split_name)
        if not os.path.isdir(split_path):
            continue

        for person in sorted(os.listdir(split_path)):
            person_path = os.path.join(split_path, person)
            if os.path.isdir(person_path):
                out.append((person, person_path))

    return out


def detect_face_or_fallback(
    image_gray: np.ndarray,
    face_cascade: cv.CascadeClassifier,
    min_face_size: int,
) -> np.ndarray | None:
    detected = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )

    if len(detected) > 0:
        x, y, w, h = max(detected, key=lambda box: box[2] * box[3])
        roi = image_gray[y : y + h, x : x + w]
    else:
        if image_gray.shape[0] < min_face_size or image_gray.shape[1] < min_face_size:
            return None
        roi = image_gray

    roi = cv.equalizeHist(roi)
    return cv.resize(roi, IMG_SIZE)


def gather_training_data(
    person_dirs: List[Tuple[str, str]],
    face_cascade: cv.CascadeClassifier,
    max_images_per_person: int,
    min_face_size: int,
) -> Tuple[List[np.ndarray], List[int], Dict[str, int], Dict[str, int], int]:
    faces: List[np.ndarray] = []
    labels: List[int] = []
    label_map: Dict[str, int] = {}
    person_counts: Dict[str, int] = defaultdict(int)

    skipped_images = 0

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

        for image_name in image_files:
            image_path = os.path.join(folder, image_name)
            img = cv.imread(image_path)
            if img is None:
                skipped_images += 1
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            face = detect_face_or_fallback(gray, face_cascade, min_face_size)
            if face is None:
                skipped_images += 1
                continue

            faces.append(face)
            labels.append(label_id)
            person_counts[person] += 1

    return faces, labels, label_map, dict(person_counts), skipped_images


def main() -> None:
    args = parse_args()

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

    print("[INFO] Initializing Haar cascade and LBPH recognizer...")
    face_cascade = cv.CascadeClassifier(args.cascade_path)
    if face_cascade.empty():
        raise FileNotFoundError(f"Could not load cascade file: {args.cascade_path}")

    recognizer = cv.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )

    print(f"[INFO] Using training dataset: {raw_root} (enabled={args.include_raw})")
    print(f"[INFO] Using processed dataset: {processed_root} (enabled={args.include_processed})")
    print(f"[INFO] Using augmented dataset: {augmented_root}")
    print(f"[INFO] Included augmented splits: {sorted(include_splits)}")

    raw_people = get_person_dirs_from_raw(raw_root) if args.include_raw else []
    processed_people = get_person_dirs_from_processed(processed_root) if args.include_processed else []
    augmented_people = get_person_dirs_from_augmented(augmented_root, include_splits)
    person_dirs = raw_people + processed_people + augmented_people

    if not person_dirs:
        raise RuntimeError(
            "No training folders found. Check base-data-dir and dataset folder names."
        )

    total_start = time.time()
    faces, labels, label_map, person_counts, skipped_images = gather_training_data(
        person_dirs=person_dirs,
        face_cascade=face_cascade,
        max_images_per_person=args.max_images_per_person,
        min_face_size=args.min_face_size,
    )

    if not faces:
        raise RuntimeError("No training faces extracted. Cannot train model.")

    print(f"[INFO] Training samples collected: {len(faces)}")
    print(f"[INFO] Identities: {len(label_map)}")
    print(f"[INFO] Skipped images: {skipped_images}")

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
