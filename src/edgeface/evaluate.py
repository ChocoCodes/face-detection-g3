import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.edgeface.common import (
    cosine,
    enhance_raw_face,
    extract_face_crop,
    gather_samples,
    l2_normalize,
    preprocess_for_edgeface,
    resolve_path,
    root_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YuNet + EdgeFace enrollment on raw, processed, and augmented datasets."
    )
    parser.add_argument("--base-data-dir", default=root_path("data"))
    parser.add_argument("--raw-dir-name", default="lasalle_db1")
    parser.add_argument("--processed-dir-name", default="lasalle_db1_processed")
    parser.add_argument("--augmented-dir-name", default="augmented41mods")
    parser.add_argument("--aug-splits", default="original,light,medium,heavy")
    parser.add_argument(
        "--yunet-model",
        default=root_path("models", "yunet_mobilefacenet", "face_detection_yunet_2023mar.onnx"),
    )
    parser.add_argument(
        "--edgeface-model",
        default=root_path("models", "edgeface", "edgeface_xs.onnx"),
    )
    parser.add_argument(
        "--enrollment-path",
        default=root_path("models", "edgeface", "enrollment.json"),
    )
    parser.add_argument(
        "--report-json",
        default=root_path("reports", "evaluation", "edgeface_eval.json"),
    )
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--det-score-threshold", type=float, default=0.6)
    parser.add_argument("--det-nms-threshold", type=float, default=0.3)
    parser.add_argument("--det-top-k", type=int, default=5000)
    parser.add_argument("--embed-input-size", type=int, default=112)
    parser.add_argument("--align-face", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--raw-fallback-full-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-detect-max-side", type=int, default=640)
    parser.add_argument("--raw-clahe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-gamma", type=float, default=1.15)
    parser.add_argument("--max-images-per-person", type=int, default=0)
    return parser.parse_args()


def bucket_summary(name: str, stats: dict[str, int]) -> dict:
    evaluated = stats["evaluated_images"]
    return {
        "bucket": name,
        "total_images": stats["total_images"],
        "evaluated_images": evaluated,
        "correct": stats["correct"],
        "hit_rate_percent": (100.0 * stats["correct"] / evaluated) if evaluated else 0.0,
        "predicted_known": stats["predicted_known"],
        "predicted_unknown": stats["predicted_unknown"],
        "face_detected": stats["face_detected"],
        "face_fallback_used": stats["face_fallback_used"],
        "skipped_unreadable": stats["skipped_unreadable"],
        "skipped_no_face": stats["skipped_no_face"],
        "skipped_unseen_identity": stats["skipped_unseen_identity"],
    }


def main() -> None:
    args = parse_args()
    args.base_data_dir = resolve_path(args.base_data_dir)
    args.yunet_model = resolve_path(args.yunet_model)
    args.edgeface_model = resolve_path(args.edgeface_model)
    args.enrollment_path = resolve_path(args.enrollment_path)
    args.report_json = resolve_path(args.report_json)

    if not os.path.exists(args.yunet_model):
        raise FileNotFoundError(f"YuNet model not found: {args.yunet_model}")
    if not os.path.exists(args.edgeface_model):
        raise FileNotFoundError(f"EdgeFace model not found: {args.edgeface_model}")
    if not os.path.exists(args.enrollment_path):
        raise FileNotFoundError(f"Enrollment file not found: {args.enrollment_path}")

    with open(args.enrollment_path, "r", encoding="utf-8") as f:
        enrollment = json.load(f)

    training_cfg = enrollment.get("metadata", {}).get("training_config", {})
    if args.threshold < 0:
        args.threshold = float(enrollment.get("metadata", {}).get("recommended_threshold", 0.55))
    if args.align_face is None:
        args.align_face = bool(training_cfg.get("align_face", True))

    centroids = {
        person: l2_normalize(np.array(payload["centroid"], dtype=np.float32))
        for person, payload in enrollment["people"].items()
    }

    detector = cv.FaceDetectorYN.create(
        args.yunet_model,
        "",
        (320, 320),
        args.det_score_threshold,
        args.det_nms_threshold,
        args.det_top_k,
    )
    embedder = cv.dnn.readNetFromONNX(args.edgeface_model)

    aug_splits = {s.strip().lower() for s in args.aug_splits.split(",") if s.strip()}
    samples = gather_samples(
        base_data_dir=args.base_data_dir,
        raw_dir=args.raw_dir_name,
        processed_dir=args.processed_dir_name,
        augmented_dir=args.augmented_dir_name,
        aug_splits=aug_splits,
        include_raw=True,
        max_images_per_person=args.max_images_per_person,
    )
    if not samples:
        raise RuntimeError("No evaluation samples found.")

    per_bucket = defaultdict(
        lambda: {
            "total_images": 0,
            "evaluated_images": 0,
            "correct": 0,
            "predicted_known": 0,
            "predicted_unknown": 0,
            "face_detected": 0,
            "face_fallback_used": 0,
            "skipped_unreadable": 0,
            "skipped_no_face": 0,
            "skipped_unseen_identity": 0,
        }
    )
    misclassified = []
    start = time.time()

    for sample in samples:
        bucket = per_bucket[sample.bucket]
        bucket["total_images"] += 1

        if sample.person not in centroids:
            bucket["skipped_unseen_identity"] += 1
            continue

        img = cv.imread(sample.path)
        if img is None:
            bucket["skipped_unreadable"] += 1
            continue

        face_crop, detected = extract_face_crop(
            img_bgr=img,
            detector=detector,
            detect_max_side=args.raw_detect_max_side if sample.bucket == "raw" else 0,
            input_size=args.embed_input_size,
            align_face=args.align_face,
        )
        if face_crop is None:
            if sample.bucket == "raw" and args.raw_fallback_full_image:
                face_crop = img
                bucket["face_fallback_used"] += 1
            else:
                bucket["skipped_no_face"] += 1
                continue
        elif detected:
            bucket["face_detected"] += 1

        if sample.bucket == "raw":
            face_crop = enhance_raw_face(face_crop, args.raw_clahe, args.raw_gamma)

        blob = preprocess_for_edgeface(face_crop, args.embed_input_size)
        embedder.setInput(blob)
        emb = l2_normalize(embedder.forward().flatten().astype(np.float32))

        best_person = "Unknown"
        best_score = -1.0
        for person, centroid in centroids.items():
            score = cosine(emb, centroid)
            if score > best_score:
                best_score = score
                best_person = person

        predicted = best_person if best_score >= args.threshold else "Unknown"
        bucket["evaluated_images"] += 1
        if predicted == "Unknown":
            bucket["predicted_unknown"] += 1
        else:
            bucket["predicted_known"] += 1
        if predicted == sample.person:
            bucket["correct"] += 1
        elif len(misclassified) < 20:
            misclassified.append(
                {
                    "bucket": sample.bucket,
                    "truth": sample.person,
                    "predicted": predicted,
                    "score": float(best_score),
                    "path": sample.path,
                }
            )

    overall = {
        key: sum(bucket[key] for bucket in per_bucket.values())
        for key in next(iter(per_bucket.values()), {})
    }
    report = {
        "model_family": "edgeface",
        "enrollment_path": args.enrollment_path,
        "threshold": args.threshold,
        "elapsed_seconds": time.time() - start,
        "buckets": [bucket_summary(name, per_bucket[name]) for name in sorted(per_bucket.keys())],
        "overall": bucket_summary("overall", overall),
        "sample_misclassifications": misclassified,
    }

    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[RESULT] YuNet + EdgeFace Evaluation")
    print(f"[INFO] Threshold: {args.threshold:.3f}")
    print(f"[INFO] Wrote JSON report to: {args.report_json}")
    if overall["evaluated_images"] > 0:
        hit_rate = 100.0 * overall["correct"] / overall["evaluated_images"]
        print(f"[INFO] Overall hit rate: {hit_rate:.2f}%")


if __name__ == "__main__":
    main()
