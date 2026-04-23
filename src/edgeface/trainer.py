import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
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
        description="Build identity enrollment using YuNet detection + EdgeFace embeddings."
    )
    parser.add_argument("--base-data-dir", default=root_path("data"))
    parser.add_argument("--raw-dir-name", default="lasalle_db1")
    parser.add_argument("--processed-dir-name", default="lasalle_db1_processed")
    parser.add_argument("--augmented-dir-name", default="augmented41mods")
    parser.add_argument("--aug-splits", default="heavy,medium,light")
    parser.add_argument("--include-raw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-processed", action="store_true")
    parser.add_argument("--max-images-per-person", type=int, default=0)
    parser.add_argument(
        "--yunet-model",
        default=root_path("models", "yunet_mobilefacenet", "face_detection_yunet_2023mar.onnx"),
    )
    parser.add_argument(
        "--edgeface-model",
        default=root_path("models", "edgeface", "edgeface_xs.onnx"),
    )
    parser.add_argument(
        "--enrollment-output",
        default=root_path("models", "edgeface", "enrollment.json"),
    )
    parser.add_argument("--det-score-threshold", type=float, default=0.6)
    parser.add_argument("--det-nms-threshold", type=float, default=0.3)
    parser.add_argument("--det-top-k", type=int, default=5000)
    parser.add_argument("--embed-input-size", type=int, default=112)
    parser.add_argument("--align-face", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-fallback-full-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-detect-max-side", type=int, default=640)
    parser.add_argument("--raw-clahe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-gamma", type=float, default=1.15)
    parser.add_argument("--raw-weight", type=float, default=2.0)
    parser.add_argument(
        "--threshold-candidates",
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_data_dir = resolve_path(args.base_data_dir)
    args.yunet_model = resolve_path(args.yunet_model)
    args.edgeface_model = resolve_path(args.edgeface_model)
    args.enrollment_output = resolve_path(args.enrollment_output)

    if not os.path.exists(args.yunet_model):
        raise FileNotFoundError(f"YuNet model not found: {args.yunet_model}")
    if not os.path.exists(args.edgeface_model):
        raise FileNotFoundError(f"EdgeFace model not found: {args.edgeface_model}")

    os.makedirs(os.path.dirname(args.enrollment_output), exist_ok=True)

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
        include_raw=args.include_raw,
        include_processed=args.include_processed,
        include_augmented=True,
        max_images_per_person=args.max_images_per_person,
    )
    if not samples:
        raise RuntimeError("No training samples found.")

    per_person_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    per_person_weights: dict[str, list[float]] = defaultdict(list)
    diagnostics = {
        "total_samples": len(samples),
        "used_samples": 0,
        "skipped_unreadable": 0,
        "skipped_no_face": 0,
        "bucket_counts": defaultdict(int),
    }

    start = time.time()
    for sample in samples:
        img = cv.imread(sample.path)
        if img is None:
            diagnostics["skipped_unreadable"] += 1
            continue

        face_crop, _ = extract_face_crop(
            img_bgr=img,
            detector=detector,
            detect_max_side=args.raw_detect_max_side if sample.bucket == "raw" else 0,
            input_size=args.embed_input_size,
            align_face=args.align_face,
        )
        if face_crop is None:
            if sample.bucket == "raw" and args.raw_fallback_full_image:
                face_crop = img
            else:
                diagnostics["skipped_no_face"] += 1
                continue

        if sample.bucket == "raw":
            face_crop = enhance_raw_face(face_crop, args.raw_clahe, args.raw_gamma)

        blob = preprocess_for_edgeface(face_crop, args.embed_input_size)
        embedder.setInput(blob)
        emb = l2_normalize(embedder.forward().flatten().astype(np.float32))

        per_person_embeddings[sample.person].append(emb)
        per_person_weights[sample.person].append(args.raw_weight if sample.bucket == "raw" else 1.0)
        diagnostics["used_samples"] += 1
        diagnostics["bucket_counts"][sample.bucket] += 1

    if not per_person_embeddings:
        raise RuntimeError("No embeddings extracted.")

    centroids: dict[str, np.ndarray] = {}
    sample_embeddings: list[tuple[str, np.ndarray]] = []
    for person, embs in per_person_embeddings.items():
        mat = np.vstack(embs)
        weights = np.array(per_person_weights[person], dtype=np.float32)
        weighted_sum = np.sum(mat * weights[:, None], axis=0)
        centroid = l2_normalize((weighted_sum / np.sum(weights)).astype(np.float32))
        centroids[person] = centroid
        for emb in embs:
            sample_embeddings.append((person, emb))

    threshold_scores = []
    candidate_thresholds = [float(x.strip()) for x in args.threshold_candidates.split(",") if x.strip()]
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
            if max_other >= -0.5 and max_other < thr:
                unknown_below += 1
            if max_other >= -0.5:
                unknown_total += 1
        acc = correct / len(sample_embeddings)
        tpr = known_above / known_total if known_total else 0.0
        tnr = unknown_below / unknown_total if unknown_total else 0.0
        threshold_scores.append(
            {
                "threshold": thr,
                "id_accuracy": acc,
                "known_accept_rate": tpr,
                "unknown_reject_rate": tnr,
                "balanced_accuracy": 0.5 * (tpr + tnr),
            }
        )

    best_threshold = max(threshold_scores, key=lambda row: row["balanced_accuracy"])["threshold"]
    enrollment = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_family": "edgeface",
            "yunet_model": args.yunet_model,
            "edgeface_model": args.edgeface_model,
            "embedding_dim": int(next(iter(centroids.values())).shape[0]),
            "recommended_threshold": float(best_threshold),
            "threshold_scores": threshold_scores,
            "diagnostics": {
                "total_samples": diagnostics["total_samples"],
                "used_samples": diagnostics["used_samples"],
                "skipped_unreadable": diagnostics["skipped_unreadable"],
                "skipped_no_face": diagnostics["skipped_no_face"],
                "bucket_counts": dict(diagnostics["bucket_counts"]),
            },
            "training_config": {
                "include_raw": args.include_raw,
                "include_processed": args.include_processed,
                "aug_splits": sorted(aug_splits),
                "raw_fallback_full_image": args.raw_fallback_full_image,
                "raw_detect_max_side": args.raw_detect_max_side,
                "raw_clahe": args.raw_clahe,
                "raw_gamma": args.raw_gamma,
                "raw_weight": args.raw_weight,
                "max_images_per_person": args.max_images_per_person,
                "embed_input_size": args.embed_input_size,
                "align_face": args.align_face,
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

    elapsed = time.time() - start
    print(f"[OK] Wrote enrollment: {args.enrollment_output}")
    print(f"[INFO] People enrolled: {len(centroids)}")
    print(f"[INFO] Used samples: {diagnostics['used_samples']} / {diagnostics['total_samples']}")
    print(f"[INFO] Recommended threshold: {best_threshold:.3f}")
    print(f"[TIME] Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
