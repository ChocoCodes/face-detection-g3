import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv

from src.dataset_layout import gather_augmented_person_dirs, infer_target_split_name
from src.lbph.preprocess import IMG_SIZE, extract_lbph_face, resolve_eye_cascade_path
from src.reporting.identity import (
    attach_entity_identity,
    build_dataset_profile,
    derive_model_variant,
)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Stats:
    total_images: int = 0
    evaluated_images: int = 0
    correct: int = 0
    known_total: int = 0
    known_correct: int = 0
    unknown_total: int = 0
    unknown_correct: int = 0
    predicted_known: int = 0
    predicted_unknown: int = 0
    face_detected: int = 0
    face_aligned: int = 0
    skipped_unreadable: int = 0
    skipped_no_face: int = 0
    skipped_too_small: int = 0


def root_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def resolve_path(path_value: str) -> str:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)
    return str(PROJECT_ROOT.joinpath(candidate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained LBPH model on split/test by default."
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
        default="light",
        help="Comma-separated augmented subsets to include.",
    )
    parser.add_argument("--model-path", default=root_path("models", "lbph", "trainer_lasalle.yml"))
    parser.add_argument("--labels-path", default=root_path("models", "lbph", "labels_lasalle.json"))
    parser.add_argument("--cascade-path", default=root_path("haar", "haarcascade_frontalface_default.xml"))
    parser.add_argument(
        "--eye-cascade-path",
        default="",
        help="Optional path to Haar eye cascade XML. If empty, OpenCV default is used.",
    )
    parser.add_argument(
        "--align-eyes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable classical eye-based alignment before equalization/resizing.",
    )
    parser.add_argument(
        "--equalization",
        choices=["equalize", "clahe"],
        default="equalize",
        help="Face contrast normalization.",
    )
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
        "--random-seed",
        type=int,
        default=42,
        help="Deterministic random seed for per-person sampling when capping images.",
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
        help="Deprecated compatibility flag; ignored for honest evaluation.",
    )
    parser.add_argument(
        "--downscale-max-side",
        type=int,
        default=0,
        help="If >0, downscale image before Haar detection and remap coordinates.",
    )
    parser.add_argument(
        "--report-json",
        default=root_path("reports", "evaluation", "lbph_eval.json"),
        help="Path to save evaluation summary as JSON.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional run tag to disambiguate reports for the same model/dataset profile.",
    )
    parser.add_argument(
        "--threshold-sweep",
        default="35,40,45,50,55,60,65,70,75,80,90,100",
        help="Comma-separated LBPH distance thresholds for sweep analysis.",
    )
    return parser.parse_args()


def is_image_file(file_name: str) -> bool:
    _, ext = os.path.splitext(file_name)
    return ext.lower() in ALLOWED_EXTENSIONS


def stable_person_seed(base_seed: int, bucket: str, person: str) -> int:
    token = f"{bucket}:{person}"
    return base_seed + sum(ord(ch) for ch in token)


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


def summarize_bucket(name: str, stats: Stats) -> str:
    eval_count = stats.evaluated_images
    overall_acc = (100.0 * stats.correct / eval_count) if eval_count else 0.0
    known_acc = (100.0 * stats.known_correct / stats.known_total) if stats.known_total else 0.0
    unknown_reject = (
        (100.0 * stats.unknown_correct / stats.unknown_total) if stats.unknown_total else 0.0
    )
    balanced = 0.5 * (known_acc + unknown_reject) if (stats.known_total and stats.unknown_total) else 0.0

    return (
        f"{name:<18} total={stats.total_images:<6} eval={eval_count:<6} "
        f"overall={overall_acc:6.2f}% known={known_acc:6.2f}% unk_rej={unknown_reject:6.2f}% "
        f"bal={balanced:6.2f}% detected={stats.face_detected:<6} aligned={stats.face_aligned:<6}"
    )


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


def print_progress_stream(processed: int, start_time: float, status: str) -> None:
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    msg = (
        f"\r[PROGRESS] processed={processed:<8} "
        f"| elapsed {format_seconds(elapsed)} "
        f"| {rate:6.2f} img/s | {status:<28}"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()


def bucket_to_dict(name: str, stats: Stats) -> dict:
    eval_count = stats.evaluated_images
    overall_acc = (100.0 * stats.correct / eval_count) if eval_count else 0.0
    known_acc = (100.0 * stats.known_correct / stats.known_total) if stats.known_total else 0.0
    unknown_reject = (
        (100.0 * stats.unknown_correct / stats.unknown_total) if stats.unknown_total else 0.0
    )
    balanced = 0.5 * (known_acc + unknown_reject) if (stats.known_total and stats.unknown_total) else 0.0

    return {
        "bucket": name,
        "total_images": stats.total_images,
        "evaluated_images": eval_count,
        "correct": stats.correct,
        "overall_accuracy_percent": overall_acc,
        "known_total": stats.known_total,
        "known_correct": stats.known_correct,
        "known_accuracy_percent": known_acc,
        "unknown_total": stats.unknown_total,
        "unknown_correct": stats.unknown_correct,
        "unknown_rejection_rate_percent": unknown_reject,
        "balanced_accuracy_percent": balanced,
        "predicted_known": stats.predicted_known,
        "predicted_unknown": stats.predicted_unknown,
        "face_detected": stats.face_detected,
        "face_aligned": stats.face_aligned,
        "skipped_unreadable": stats.skipped_unreadable,
        "skipped_no_face": stats.skipped_no_face,
        "skipped_too_small": stats.skipped_too_small,
    }


def threshold_metrics(records: list[dict], threshold: float) -> dict:
    total = len(records)
    known_records = [r for r in records if r["is_known_truth"]]
    unknown_records = [r for r in records if not r["is_known_truth"]]

    known_total = len(known_records)
    unknown_total = len(unknown_records)

    known_correct = 0
    unknown_correct = 0
    overall_correct = 0

    for r in records:
        dist = float(r["distance"])
        best_name = r["best_name"]
        pred = best_name if (best_name != "Unknown" and dist <= threshold) else "Unknown"

        if r["is_known_truth"]:
            if pred == r["truth"]:
                known_correct += 1
                overall_correct += 1
        else:
            if pred == "Unknown":
                unknown_correct += 1
                overall_correct += 1

    known_acc = (100.0 * known_correct / known_total) if known_total else 0.0
    unknown_reject = (100.0 * unknown_correct / unknown_total) if unknown_total else 0.0
    balanced = 0.5 * (known_acc + unknown_reject) if (known_total and unknown_total) else 0.0
    overall_acc = (100.0 * overall_correct / total) if total else 0.0

    return {
        "threshold": threshold,
        "overall_total": total,
        "overall_correct": overall_correct,
        "overall_accuracy_percent": overall_acc,
        "known_total": known_total,
        "known_correct": known_correct,
        "known_accuracy_percent": known_acc,
        "unknown_total": unknown_total,
        "unknown_correct": unknown_correct,
        "unknown_rejection_rate_percent": unknown_reject,
        "balanced_accuracy_percent": balanced,
    }


def compute_threshold_sweep(eval_records: list[dict], thresholds: list[float]) -> list[dict]:
    sweep = []
    buckets = sorted({r["bucket"] for r in eval_records})

    for thr in thresholds:
        overall = threshold_metrics(eval_records, thr)
        by_bucket = {}
        for bucket in buckets:
            bucket_records = [r for r in eval_records if r["bucket"] == bucket]
            by_bucket[bucket] = threshold_metrics(bucket_records, thr)
        sweep.append({
            **overall,
            "by_bucket": by_bucket,
        })

    return sweep


def canonical_source_stem(file_name: str) -> str:
    stem = Path(file_name).stem.lower()
    stem = re.sub(r"(__|_aug|_light|_medium|_heavy|-aug).*", "", stem)
    return stem


def estimate_augmented_overlap(entries: list[tuple[str, str, str]]) -> tuple[int, int]:
    base_stems: dict[str, set[str]] = defaultdict(set)
    aug_stems: dict[str, set[str]] = defaultdict(set)

    for bucket, person, person_dir in entries:
        if not os.path.isdir(person_dir):
            continue
        files = [f for f in os.listdir(person_dir) if is_image_file(f)]
        stems = {canonical_source_stem(f) for f in files}
        if bucket == "augmented":
            aug_stems[person].update(stems)
        else:
            base_stems[person].update(stems)

    overlap_people = 0
    overlap_stems = 0
    for person, stems in aug_stems.items():
        inter = stems.intersection(base_stems.get(person, set()))
        if inter:
            overlap_people += 1
            overlap_stems += len(inter)

    return overlap_people, overlap_stems


def main() -> None:
    args = parse_args()
    args.base_data_dir = resolve_path(args.base_data_dir)
    args.model_path = resolve_path(args.model_path)
    args.labels_path = resolve_path(args.labels_path)
    args.cascade_path = resolve_path(args.cascade_path)
    args.eye_cascade_path = resolve_path(resolve_eye_cascade_path(args.eye_cascade_path))
    if args.report_json:
        args.report_json = resolve_path(args.report_json)

    with open(args.labels_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    id_to_name = {int(v): k for k, v in label_map.items()}
    known_names = set(label_map.keys())

    face_cascade = cv.CascadeClassifier(args.cascade_path)
    if face_cascade.empty():
        raise FileNotFoundError(f"Could not load cascade file: {args.cascade_path}")

    eye_cascade: cv.CascadeClassifier | None = None
    if args.align_eyes:
        eye_cascade = cv.CascadeClassifier(args.eye_cascade_path)
        if eye_cascade.empty():
            print(
                f"[WARN] Could not load eye cascade at {args.eye_cascade_path}. "
                "Alignment will be disabled."
            )
            eye_cascade = None

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

    if args.assume_processed_are_cropped:
        print(
            "[WARN] --assume-processed-are-cropped is deprecated and ignored to avoid "
            "full-image contamination."
        )

    if args.include_augmented:
        print("[WARN] Augmented evaluation is enabled and may inflate metrics.")
        if aug_splits != {"light"}:
            print("[WARN] Recommended default for LBPH evaluation is light-only augmentation.")

    overlap_people, overlap_stems = estimate_augmented_overlap(entries)
    leakage_warning = None
    if args.include_augmented and overlap_people > 0:
        leakage_warning = (
            f"Possible source overlap detected for {overlap_people} identities "
            f"({overlap_stems} canonical stems)."
        )
        print(f"[WARN] {leakage_warning}")

    per_bucket_stats = defaultdict(Stats)
    overall = Stats()
    misclassified = []
    eval_records: list[dict] = []

    print(
        f"[INFO] Evaluating {len(entries)} identity folders "
        "(streaming mode: no long pre-count pass)"
    )

    start_time = time.time()
    processed_images = 0
    progress_interval = 25

    for bucket_name, person_name, person_path in entries:
        image_files = [f for f in sorted(os.listdir(person_path)) if is_image_file(f)]
        if args.max_images_per_person > 0:
            rng = random.Random(stable_person_seed(args.random_seed, bucket_name, person_name))
            sampled = list(image_files)
            rng.shuffle(sampled)
            image_files = sampled[: args.max_images_per_person]

        for image_name in image_files:
            processed_images += 1
            if processed_images == 1 or processed_images % progress_interval == 0:
                print_progress_stream(
                    processed=processed_images,
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

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            processed = extract_lbph_face(
                image_gray=gray,
                face_cascade=face_cascade,
                min_face_size=args.min_face_size,
                scale_factor=args.scale_factor,
                min_neighbors=args.min_neighbors,
                img_size=IMG_SIZE,
                equalization=args.equalization,
                align_eyes=args.align_eyes,
                eye_cascade=eye_cascade,
                downscale_max_side=args.downscale_max_side,
            )
            if processed.face is None:
                if processed.reason == "image_too_small":
                    per_bucket_stats[bucket_name].skipped_too_small += 1
                    overall.skipped_too_small += 1
                else:
                    per_bucket_stats[bucket_name].skipped_no_face += 1
                    overall.skipped_no_face += 1
                continue

            per_bucket_stats[bucket_name].evaluated_images += 1
            overall.evaluated_images += 1
            per_bucket_stats[bucket_name].face_detected += 1
            overall.face_detected += 1
            if processed.used_alignment:
                per_bucket_stats[bucket_name].face_aligned += 1
                overall.face_aligned += 1

            pred_id, distance = recognizer.predict(processed.face)
            pred_name = id_to_name.get(pred_id, "Unknown")

            if distance <= args.unknown_threshold and pred_name != "Unknown":
                predicted_label = pred_name
                per_bucket_stats[bucket_name].predicted_known += 1
                overall.predicted_known += 1
            else:
                predicted_label = "Unknown"
                per_bucket_stats[bucket_name].predicted_unknown += 1
                overall.predicted_unknown += 1

            is_known_truth = person_name in known_names
            if is_known_truth:
                per_bucket_stats[bucket_name].known_total += 1
                overall.known_total += 1
                is_correct = predicted_label == person_name
                if is_correct:
                    per_bucket_stats[bucket_name].known_correct += 1
                    overall.known_correct += 1
            else:
                per_bucket_stats[bucket_name].unknown_total += 1
                overall.unknown_total += 1
                is_correct = predicted_label == "Unknown"
                if is_correct:
                    per_bucket_stats[bucket_name].unknown_correct += 1
                    overall.unknown_correct += 1

            if is_correct:
                per_bucket_stats[bucket_name].correct += 1
                overall.correct += 1
            elif len(misclassified) < args.show_misclassified:
                misclassified.append(
                    {
                        "bucket": bucket_name,
                        "truth": person_name,
                        "truth_known": is_known_truth,
                        "predicted": predicted_label,
                        "best_name": pred_name,
                        "distance": float(distance),
                        "path": image_path,
                    }
                )

            eval_records.append(
                {
                    "bucket": bucket_name,
                    "truth": person_name,
                    "is_known_truth": is_known_truth,
                    "best_name": pred_name,
                    "distance": float(distance),
                }
            )

    elapsed = time.time() - start_time

    if processed_images > 0:
        print_progress_stream(
            processed=processed_images,
            start_time=start_time,
            status="done",
        )
        print()

    print("\n[RESULT] LBPH Offline Evaluation")
    print(f"[INFO] Model: {args.model_path}")
    print(f"[INFO] Labels: {args.labels_path}")
    print(f"[INFO] Unknown threshold: {args.unknown_threshold}")
    print(f"[INFO] Evaluated in: {elapsed:.2f}s")
    print(
        f"[INFO] Preprocess: detect->align({args.align_eyes})->{args.equalization}->resize{IMG_SIZE}"
    )

    print("\n[METRICS BY DATA BUCKET]")
    for bucket_name in sorted(per_bucket_stats.keys()):
        print(summarize_bucket(bucket_name, per_bucket_stats[bucket_name]))

    print("\n[OVERALL]")
    print(summarize_bucket("overall", overall))

    print("\n[SKIPS]")
    print(f"Unreadable images: {overall.skipped_unreadable}")
    print(f"Skipped no-face: {overall.skipped_no_face}")
    print(f"Too small / unusable: {overall.skipped_too_small}")

    if misclassified:
        print("\n[SAMPLE MISCLASSIFICATIONS]")
        for item in misclassified:
            print(
                f"{item['bucket']}: truth={item['truth']} pred={item['predicted']} "
                f"best={item['best_name']} dist={item['distance']:.2f} path={item['path']}"
            )

    thresholds = [float(x.strip()) for x in args.threshold_sweep.split(",") if x.strip()]
    if not thresholds:
        thresholds = [35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 90.0, 100.0]
    threshold_sweep = compute_threshold_sweep(eval_records, thresholds)

    if args.report_json:
        config = {
            "unknown_threshold": args.unknown_threshold,
            "threshold_sweep": ",".join(
                str(int(t)) if float(t).is_integer() else str(t) for t in thresholds
            ),
            "align_eyes": args.align_eyes,
            "equalization": args.equalization,
            "min_face_size": args.min_face_size,
            "scale_factor": args.scale_factor,
            "min_neighbors": args.min_neighbors,
            "downscale_max_side": args.downscale_max_side,
            "max_images_per_person": args.max_images_per_person,
            "random_seed": args.random_seed,
            "include_processed": args.include_processed,
            "include_augmented": args.include_augmented,
            "aug_splits": sorted(aug_splits),
        }

        report = {
            "model_path": args.model_path,
            "labels_path": args.labels_path,
            "elapsed_seconds": elapsed,
            "config": config,
            "buckets": [
                bucket_to_dict(bucket_name, per_bucket_stats[bucket_name])
                for bucket_name in sorted(per_bucket_stats.keys())
            ],
            "overall": bucket_to_dict("overall", overall),
            "threshold_sweep": threshold_sweep,
            "known_vs_unknown": {
                "known_total": overall.known_total,
                "known_correct": overall.known_correct,
                "known_accuracy_percent": (
                    (100.0 * overall.known_correct / overall.known_total)
                    if overall.known_total
                    else 0.0
                ),
                "unknown_total": overall.unknown_total,
                "unknown_correct": overall.unknown_correct,
                "unknown_rejection_rate_percent": (
                    (100.0 * overall.unknown_correct / overall.unknown_total)
                    if overall.unknown_total
                    else 0.0
                ),
                "balanced_accuracy_percent": (
                    0.5
                    * (
                        (100.0 * overall.known_correct / overall.known_total)
                        + (100.0 * overall.unknown_correct / overall.unknown_total)
                    )
                    if (overall.known_total and overall.unknown_total)
                    else 0.0
                ),
            },
            "sample_misclassifications": misclassified,
            "skipped_reasons": {
                "unreadable": overall.skipped_unreadable,
                "no_face": overall.skipped_no_face,
                "too_small": overall.skipped_too_small,
            },
            "data_hygiene": {
                "augmented_included": args.include_augmented,
                "possible_overlap_warning": leakage_warning,
                "limitation": (
                    "Strict leakage proof is not possible from folder structure and filenames alone; "
                    "overlap is heuristic unless source metadata is provided."
                ),
            },
        }
        target_split = infer_target_split_name(
            raw_dir=args.raw_dir_name,
            processed_dir=args.processed_dir_name,
        )
        dataset_profile = build_dataset_profile(
            base_data_dir=args.base_data_dir,
            raw_dir_name=args.raw_dir_name,
            include_raw=args.raw_dir_name != "__disabled__",
            processed_dir_name=args.processed_dir_name,
            include_processed=args.include_processed,
            augmented_dir_name=args.augmented_dir_name,
            include_augmented=args.include_augmented,
            aug_splits=aug_splits,
            target_split=target_split,
        )
        model_variant = derive_model_variant(args.model_path, args.labels_path, fallback="lbph")
        attach_entity_identity(
            report=report,
            model_family="lbph",
            dataset_profile=dataset_profile,
            model_variant=model_variant,
            run_tag=args.run_tag,
        )

        os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[OK] Wrote JSON report to: {args.report_json}")


if __name__ == "__main__":
    main()
