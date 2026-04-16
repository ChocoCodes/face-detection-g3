import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained LBPH model on raw dataset by default."
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
    parser.add_argument(
        "--aug-splits",
        default="original,light,medium,heavy",
        help="Comma-separated augmented subsets to include.",
    )
    parser.add_argument("--model-path", default=root_path("models", "lbph", "trainer_lasalle.yml"))
    parser.add_argument("--labels-path", default=root_path("models", "lbph", "labels_lasalle.json"))
    parser.add_argument("--cascade-path", default=root_path("haar", "haarcascade_frontalface_default.xml"))
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=55.0,
        help="LBPH distance threshold. Lower = stricter.",
    )
    parser.add_argument("--min-face-size", type=int, default=40)
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument(
        "--max-images-per-person",
        type=int,
        default=0,
        help="Optional cap per person per dataset bucket (0 = no cap).",
    )
    parser.add_argument(
        "--show-misclassified",
        type=int,
        default=15,
        help="Show up to N misclassified samples.",
    )
    parser.add_argument(
        "--assume-processed-are-cropped",
        action="store_true",
        help="Skip face detection for processed and augmented buckets, use full image ROI directly.",
    )
    parser.add_argument(
        "--downscale-max-side",
        type=int,
        default=0,
        help="If >0, downscale image before Haar detection so longest side is this value.",
    )
    parser.add_argument(
        "--report-json",
        default=root_path("reports", "evaluation", "lbph_eval.json"),
        help="Path to save evaluation summary as JSON (overwrites existing file).",
    )
    return parser.parse_args()


def is_image_file(file_name: str) -> bool:
    _, ext = os.path.splitext(file_name)
    return ext.lower() in ALLOWED_EXTENSIONS


def detect_face_or_fallback(
    image_gray,
    face_cascade: cv.CascadeClassifier,
    min_face_size: int,
    scale_factor: float,
    min_neighbors: int,
):
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


def preprocess_direct_full_image(image_gray, min_face_size: int):
    if image_gray.shape[0] < min_face_size or image_gray.shape[1] < min_face_size:
        return None
    roi = cv.equalizeHist(image_gray)
    return cv.resize(roi, IMG_SIZE)


def maybe_downscale(gray, max_side: int):
    if max_side <= 0:
        return gray
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return gray
    scale = max_side / float(longest)
    return cv.resize(gray, (int(w * scale), int(h * scale)))


def gather_dataset_entries(
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
                split_path = os.path.join(augmented_root, split_name)
                if not os.path.isdir(split_path):
                    continue
                if aug_splits and split_name.lower() not in aug_splits:
                    continue

                bucket_name = f"augmented/{split_name}"
                for person in sorted(os.listdir(split_path)):
                    person_path = os.path.join(split_path, person)
                    if os.path.isdir(person_path):
                        entries.append((bucket_name, person, person_path))

    return entries


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


def main() -> None:
    args = parse_args()
    args.base_data_dir = resolve_path(args.base_data_dir)
    args.model_path = resolve_path(args.model_path)
    args.labels_path = resolve_path(args.labels_path)
    args.cascade_path = resolve_path(args.cascade_path)
    if args.report_json:
        args.report_json = resolve_path(args.report_json)

    with open(args.labels_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    id_to_name = {int(v): k for k, v in label_map.items()}
    known_names = set(label_map.keys())

    face_cascade = cv.CascadeClassifier(args.cascade_path)
    if face_cascade.empty():
        raise FileNotFoundError(f"Could not load cascade file: {args.cascade_path}")

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read(args.model_path)

    aug_splits = {s.strip().lower() for s in args.aug_splits.split(",") if s.strip()}
    entries = gather_dataset_entries(
        base_data_dir=args.base_data_dir,
        raw_dir=args.raw_dir_name,
        processed_dir=args.processed_dir_name,
        aug_dir=args.augmented_dir_name,
        aug_splits=aug_splits,
        include_processed=args.include_processed,
        include_augmented=args.include_augmented,
    )

    if not entries:
        raise RuntimeError("No dataset folders found to evaluate.")

    total_planned_images = 0
    for bucket_name, person_name, person_path in entries:
        image_files = [f for f in sorted(os.listdir(person_path)) if is_image_file(f)]
        if args.max_images_per_person > 0:
            image_files = image_files[: args.max_images_per_person]
        total_planned_images += len(image_files)

    per_bucket_stats = defaultdict(Stats)
    overall = Stats()
    per_person_used = defaultdict(int)
    misclassified = []

    start_time = time.time()
    processed_images = 0
    progress_interval = 25

    for bucket_name, person_name, person_path in entries:
        image_files = [f for f in sorted(os.listdir(person_path)) if is_image_file(f)]

        for image_name in image_files:
            key = (bucket_name, person_name)
            if args.max_images_per_person > 0 and per_person_used[key] >= args.max_images_per_person:
                continue
            per_person_used[key] += 1

            processed_images += 1
            if (
                processed_images == 1
                or processed_images % progress_interval == 0
                or processed_images == total_planned_images
            ):
                print_progress(
                    processed=processed_images,
                    total=total_planned_images,
                    start_time=start_time,
                    status=f"{bucket_name}/{person_name}",
                )

            image_path = os.path.join(person_path, image_name)
            per_bucket_stats[bucket_name].total_images += 1
            overall.total_images += 1

            img = cv.imread(image_path)
            if img is None:
                per_bucket_stats[bucket_name].skipped_unreadable += 1
                overall.skipped_unreadable += 1
                continue

            if person_name not in known_names:
                per_bucket_stats[bucket_name].skipped_unseen_identity += 1
                overall.skipped_unseen_identity += 1
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            if args.assume_processed_are_cropped and bucket_name != "raw":
                face = preprocess_direct_full_image(gray, args.min_face_size)
                used_fallback = True
            else:
                detect_gray = maybe_downscale(gray, args.downscale_max_side)
                face, used_fallback = detect_face_or_fallback(
                    image_gray=detect_gray,
                    face_cascade=face_cascade,
                    min_face_size=args.min_face_size,
                    scale_factor=args.scale_factor,
                    min_neighbors=args.min_neighbors,
                )

            if face is None:
                per_bucket_stats[bucket_name].skipped_too_small += 1
                overall.skipped_too_small += 1
                continue

            per_bucket_stats[bucket_name].evaluated_images += 1
            overall.evaluated_images += 1

            if used_fallback:
                per_bucket_stats[bucket_name].face_fallback_used += 1
                overall.face_fallback_used += 1
            else:
                per_bucket_stats[bucket_name].face_detected += 1
                overall.face_detected += 1

            pred_id, distance = recognizer.predict(face)
            pred_name = id_to_name.get(pred_id, "Unknown")

            if distance <= args.unknown_threshold and pred_name != "Unknown":
                predicted_label = pred_name
                per_bucket_stats[bucket_name].predicted_known += 1
                overall.predicted_known += 1
            else:
                predicted_label = "Unknown"
                per_bucket_stats[bucket_name].predicted_unknown += 1
                overall.predicted_unknown += 1

            is_correct = predicted_label == person_name
            if is_correct:
                per_bucket_stats[bucket_name].correct += 1
                overall.correct += 1
            elif len(misclassified) < args.show_misclassified:
                misclassified.append(
                    (bucket_name, person_name, predicted_label, float(distance), image_path)
                )

    elapsed = time.time() - start_time

    if total_planned_images > 0:
        print_progress(
            processed=min(processed_images, total_planned_images),
            total=total_planned_images,
            start_time=start_time,
            status="done",
        )
        print()

    print("\n[RESULT] LBPH Offline Evaluation")
    print(f"[INFO] Model: {args.model_path}")
    print(f"[INFO] Labels: {args.labels_path}")
    print(f"[INFO] Unknown threshold: {args.unknown_threshold}")
    print(f"[INFO] Evaluated in: {elapsed:.2f}s")

    print("\n[HIT RATE BY DATA BUCKET]")
    for bucket_name in sorted(per_bucket_stats.keys()):
        print(summarize_bucket(bucket_name, per_bucket_stats[bucket_name]))

    print("\n[OVERALL]")
    print(summarize_bucket("overall", overall))

    if overall.evaluated_images > 0:
        overall_hit = 100.0 * overall.correct / overall.evaluated_images
        print(f"Overall hit rate: {overall_hit:.2f}%")

    print("\n[SKIPS]")
    print(f"Unreadable images: {overall.skipped_unreadable}")
    print(f"Too small / unusable: {overall.skipped_too_small}")
    print(f"Unseen identities: {overall.skipped_unseen_identity}")

    if misclassified:
        print("\n[SAMPLE MISCLASSIFICATIONS]")
        for bucket_name, truth, pred, distance, image_path in misclassified:
            print(
                f"{bucket_name}: truth={truth} pred={pred} dist={distance:.2f} path={image_path}"
            )

    if args.report_json:
        report = {
            "model_path": args.model_path,
            "labels_path": args.labels_path,
            "unknown_threshold": args.unknown_threshold,
            "elapsed_seconds": elapsed,
            "buckets": [
                bucket_to_dict(bucket_name, per_bucket_stats[bucket_name])
                for bucket_name in sorted(per_bucket_stats.keys())
            ],
            "overall": bucket_to_dict("overall", overall),
            "sample_misclassifications": [
                {
                    "bucket": bucket_name,
                    "truth": truth,
                    "predicted": pred,
                    "distance": distance,
                    "path": image_path,
                }
                for bucket_name, truth, pred, distance, image_path in misclassified
            ],
        }

        os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[OK] Wrote JSON report to: {args.report_json}")


if __name__ == "__main__":
    main()
