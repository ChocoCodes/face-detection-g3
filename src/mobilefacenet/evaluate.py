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

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ArcFace-style 5-point template for 112x112 alignment.
REFERENCE_FIVE_POINTS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def root_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def resolve_path(path_value: str) -> str:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)
    return str(PROJECT_ROOT.joinpath(candidate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YuNet + MobileFaceNet enrollment on raw, processed, and augmented datasets."
    )
    parser.add_argument("--base-data-dir", default=root_path("data"))
    parser.add_argument("--raw-dir-name", default="lasalle_db1")
    parser.add_argument("--processed-dir-name", default="lasalle_db1_processed")
    parser.add_argument("--augmented-dir-name", default="augmented41mods")
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help="Include processed dataset in evaluation.",
    )
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="Include augmented dataset in evaluation.",
    )
    parser.add_argument("--aug-splits", default="original,light,medium,heavy")
    parser.add_argument(
        "--yunet-model",
        default=root_path("models", "yunet_mobilefacenet", "face_detection_yunet_2023mar.onnx"),
    )
    parser.add_argument(
        "--mobilefacenet-model",
        default=root_path("models", "yunet_mobilefacenet", "mobilefacenet.onnx"),
    )
    parser.add_argument(
        "--enrollment-path",
        default=root_path("models", "yunet_mobilefacenet", "enrollment.json"),
    )
    parser.add_argument(
        "--report-json",
        default=root_path("reports", "evaluation", "yunet_mobilefacenet_eval.json"),
    )
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--det-score-threshold", type=float, default=0.6)
    parser.add_argument("--det-nms-threshold", type=float, default=0.3)
    parser.add_argument("--det-top-k", type=int, default=5000)
    parser.add_argument("--embed-input-size", type=int, default=112)
    parser.add_argument(
        "--preprocess-mode",
        choices=["auto", "arcface", "mobilefacenet"],
        default="auto",
        help="Input normalization contract for embedding model. 'auto' uses enrollment config.",
    )
    parser.add_argument(
        "--align-face",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use YuNet landmarks for affine face alignment before embedding. Default follows enrollment config.",
    )
    parser.add_argument(
        "--expected-embedding-dim",
        type=int,
        default=0,
        help="Optional embedding dim contract check (0 uses enrollment metadata).",
    )
    parser.add_argument("--max-images-per-person", type=int, default=0)
    parser.add_argument(
        "--assume-processed-are-cropped",
        action="store_true",
        help="Skip face detection for processed and augmented buckets.",
    )
    parser.add_argument(
        "--raw-fallback-full-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When raw detection fails, fallback to full image crop instead of skipping.",
    )
    parser.add_argument(
        "--raw-detect-max-side",
        type=int,
        default=640,
        help="Downscale raw frame before YuNet detection for speed (0 disables).",
    )
    parser.add_argument(
        "--raw-clahe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply CLAHE normalization to raw face crops before embedding.",
    )
    parser.add_argument(
        "--raw-gamma",
        type=float,
        default=1.15,
        help="Gamma correction for raw face crops (1.0 disables).",
    )
    parser.add_argument(
        "--threshold-sweep",
        default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75",
        help="Comma-separated thresholds for post-run sweep analysis.",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-chunksize", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--show-misclassified", type=int, default=20)
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
    face_fallback_used: int = 0
    skipped_unreadable: int = 0
    skipped_no_face: int = 0
    skipped_unseen_identity: int = 0
    skipped_embed_dim_mismatch: int = 0


@dataclass
class EvalSample:
    bucket: str
    person: str
    path: str


_EVAL_DETECTOR = None
_EVAL_EMBEDDER = None
_EVAL_CENTROIDS = {}
_EVAL_PEOPLE = []
_EVAL_THRESHOLD = 0.55
_EVAL_ASSUME_PROCESSED = False
_EVAL_RAW_FALLBACK_FULL_IMAGE = True
_EVAL_RAW_DETECT_MAX_SIDE = 640
_EVAL_RAW_CLAHE = True
_EVAL_RAW_GAMMA = 1.15
_EVAL_EMBED_INPUT_SIZE = 112
_EVAL_PREPROCESS_MODE = "arcface"
_EVAL_ALIGN_FACE = True
_EVAL_EXPECTED_EMBED_DIM = 0


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


def crop_by_yunet(
    img_bgr: np.ndarray,
    detector: cv.FaceDetectorYN,
    detect_max_side: int,
):
    h, w = img_bgr.shape[:2]
    detect_img = img_bgr
    scale = 1.0

    if detect_max_side > 0:
        longest = max(h, w)
        if longest > detect_max_side:
            scale = detect_max_side / float(longest)
            detect_img = cv.resize(
                img_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv.INTER_AREA,
            )

    dh, dw = detect_img.shape[:2]
    detector.setInputSize((dw, dh))
    _, faces = detector.detect(detect_img)
    if faces is None or len(faces) == 0:
        return None

    best = max(faces, key=lambda row: float(row[14]))
    x, y, bw, bh = best[:4]
    x1 = int(max(0, x / scale))
    y1 = int(max(0, y / scale))
    x2 = int(min(w, (x + bw) / scale))
    y2 = int(min(h, (y + bh) / scale))
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]


def align_face_by_landmarks(
    img_bgr: np.ndarray,
    landmarks_xy: np.ndarray,
    output_size: int,
) -> np.ndarray | None:
    if landmarks_xy.shape != (5, 2):
        return None

    dst = REFERENCE_FIVE_POINTS_112.copy()
    if output_size != 112:
        dst = dst * (output_size / 112.0)

    matrix, _ = cv.estimateAffinePartial2D(landmarks_xy, dst, method=cv.LMEDS)
    if matrix is None:
        return None

    return cv.warpAffine(
        img_bgr,
        matrix,
        (output_size, output_size),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )


def preprocess_for_mobilefacenet(
    face_bgr: np.ndarray,
    input_size: int,
    preprocess_mode: str,
) -> np.ndarray:
    face = cv.resize(face_bgr, (input_size, input_size), interpolation=cv.INTER_AREA)
    if preprocess_mode == "arcface":
        face_rgb = cv.cvtColor(face, cv.COLOR_BGR2RGB).astype(np.float32)
        face_rgb = (face_rgb - 127.5) / 128.0
        chw = np.transpose(face_rgb, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    return cv.dnn.blobFromImage(
        face,
        scalefactor=1.0 / 128.0,
        size=(input_size, input_size),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )


def enhance_raw_face(face_bgr: np.ndarray, use_clahe: bool, gamma: float) -> np.ndarray:
    out = face_bgr

    if use_clahe:
        ycrcb = cv.cvtColor(out, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(ycrcb)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y = clahe.apply(y)
        out = cv.cvtColor(cv.merge((y, cr, cb)), cv.COLOR_YCrCb2BGR)

    if gamma > 0 and abs(gamma - 1.0) > 1e-6:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)],
            dtype=np.uint8,
        )
        out = cv.LUT(out, table)

    return out


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= 1e-12:
        return v
    return v / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _init_eval_worker(
    yunet_model: str,
    mobilefacenet_model: str,
    det_score_threshold: float,
    det_nms_threshold: float,
    det_top_k: int,
    centroids_payload: dict,
    threshold: float,
    assume_processed_are_cropped: bool,
    raw_fallback_full_image: bool,
    raw_detect_max_side: int,
    raw_clahe: bool,
    raw_gamma: float,
    embed_input_size: int,
    preprocess_mode: str,
    align_face: bool,
    expected_embed_dim: int,
) -> None:
    global _EVAL_DETECTOR
    global _EVAL_EMBEDDER
    global _EVAL_CENTROIDS
    global _EVAL_PEOPLE
    global _EVAL_THRESHOLD
    global _EVAL_ASSUME_PROCESSED
    global _EVAL_RAW_FALLBACK_FULL_IMAGE
    global _EVAL_RAW_DETECT_MAX_SIDE
    global _EVAL_RAW_CLAHE
    global _EVAL_RAW_GAMMA
    global _EVAL_EMBED_INPUT_SIZE
    global _EVAL_PREPROCESS_MODE
    global _EVAL_ALIGN_FACE
    global _EVAL_EXPECTED_EMBED_DIM

    cv.setNumThreads(1)
    _EVAL_DETECTOR = cv.FaceDetectorYN.create(
        yunet_model,
        "",
        (320, 320),
        det_score_threshold,
        det_nms_threshold,
        det_top_k,
    )
    _EVAL_EMBEDDER = cv.dnn.readNetFromONNX(mobilefacenet_model)
    _EVAL_CENTROIDS = {
        person: l2_normalize(np.array(vec, dtype=np.float32))
        for person, vec in centroids_payload.items()
    }
    _EVAL_PEOPLE = sorted(_EVAL_CENTROIDS.keys())
    _EVAL_THRESHOLD = threshold
    _EVAL_ASSUME_PROCESSED = assume_processed_are_cropped
    _EVAL_RAW_FALLBACK_FULL_IMAGE = raw_fallback_full_image
    _EVAL_RAW_DETECT_MAX_SIDE = raw_detect_max_side
    _EVAL_RAW_CLAHE = raw_clahe
    _EVAL_RAW_GAMMA = raw_gamma
    _EVAL_EMBED_INPUT_SIZE = embed_input_size
    _EVAL_PREPROCESS_MODE = preprocess_mode
    _EVAL_ALIGN_FACE = align_face
    _EVAL_EXPECTED_EMBED_DIM = expected_embed_dim


def _process_eval_sample(sample: EvalSample) -> dict:
    result = {
        "bucket": sample.bucket,
        "truth": sample.person,
        "status": "",
        "used_fallback": False,
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

    if _EVAL_ASSUME_PROCESSED and sample.bucket != "raw":
        face_crop = img
        result["used_fallback"] = True
    else:
        detect_max_side = _EVAL_RAW_DETECT_MAX_SIDE if sample.bucket == "raw" else 0
        h, w = img.shape[:2]
        detect_img = img
        scale = 1.0
        if detect_max_side > 0:
            longest = max(h, w)
            if longest > detect_max_side:
                scale = detect_max_side / float(longest)
                detect_img = cv.resize(
                    img,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv.INTER_AREA,
                )

        dh, dw = detect_img.shape[:2]
        _EVAL_DETECTOR.setInputSize((dw, dh))
        _, faces = _EVAL_DETECTOR.detect(detect_img)

        aligned_face = None
        face_crop = None
        if faces is not None and len(faces) > 0:
            best = max(faces, key=lambda row: float(row[14]))
            if _EVAL_ALIGN_FACE and len(best) >= 14:
                landmarks = np.array(best[4:14], dtype=np.float32).reshape(5, 2)
                landmarks /= scale
                aligned_face = align_face_by_landmarks(img, landmarks, _EVAL_EMBED_INPUT_SIZE)

            x, y, bw, bh = best[:4]
            x1 = int(max(0, x / scale))
            y1 = int(max(0, y / scale))
            x2 = int(min(w, (x + bw) / scale))
            y2 = int(min(h, (y + bh) / scale))
            if x2 > x1 and y2 > y1:
                face_crop = img[y1:y2, x1:x2]

        if aligned_face is not None:
            face_crop = aligned_face

        if face_crop is None:
            if sample.bucket == "raw" and _EVAL_RAW_FALLBACK_FULL_IMAGE:
                face_crop = img
                result["used_fallback"] = True
            else:
                result["status"] = "no_face"
                return result

    if sample.bucket == "raw":
        face_crop = enhance_raw_face(face_crop, _EVAL_RAW_CLAHE, _EVAL_RAW_GAMMA)

    blob = preprocess_for_mobilefacenet(face_crop, _EVAL_EMBED_INPUT_SIZE, _EVAL_PREPROCESS_MODE)
    _EVAL_EMBEDDER.setInput(blob)
    emb = l2_normalize(_EVAL_EMBEDDER.forward().flatten().astype(np.float32))
    if _EVAL_EXPECTED_EMBED_DIM > 0 and emb.shape[0] != _EVAL_EXPECTED_EMBED_DIM:
        result["status"] = "embed_dim_mismatch"
        return result

    best_person = "Unknown"
    best_score = -1.0
    for person in _EVAL_PEOPLE:
        score = cosine(emb, _EVAL_CENTROIDS[person])
        if score > best_score:
            best_score = score
            best_person = person

    predicted = best_person if best_score >= _EVAL_THRESHOLD else "Unknown"
    result["status"] = "evaluated"
    result["best_person"] = best_person
    result["predicted"] = predicted
    result["score"] = float(best_score)
    return result


def _process_eval_batch(batch_samples: List[EvalSample]) -> List[dict]:
    return [_process_eval_sample(sample) for sample in batch_samples]


def make_batches(samples: List[EvalSample], batch_size: int) -> List[List[EvalSample]]:
    size = max(1, batch_size)
    return [samples[i : i + size] for i in range(0, len(samples), size)]


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
        "skipped_no_face": stats.skipped_no_face,
        "skipped_unseen_identity": stats.skipped_unseen_identity,
        "skipped_embed_dim_mismatch": stats.skipped_embed_dim_mismatch,
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
    args.yunet_model = resolve_path(args.yunet_model)
    args.mobilefacenet_model = resolve_path(args.mobilefacenet_model)
    args.enrollment_path = resolve_path(args.enrollment_path)
    args.report_json = resolve_path(args.report_json)

    if not os.path.exists(args.yunet_model):
        raise FileNotFoundError(f"YuNet model not found: {args.yunet_model}")
    if not os.path.exists(args.mobilefacenet_model):
        raise FileNotFoundError(f"MobileFaceNet model not found: {args.mobilefacenet_model}")
    if not os.path.exists(args.enrollment_path):
        raise FileNotFoundError(f"Enrollment file not found: {args.enrollment_path}")

    with open(args.enrollment_path, "r", encoding="utf-8") as f:
        enrollment = json.load(f)

    people = sorted(enrollment["people"].keys())
    centroids_payload = {
        person: enrollment["people"][person]["centroid"]
        for person in people
    }

    if args.threshold < 0:
        args.threshold = float(enrollment["metadata"].get("recommended_threshold", 0.55))

    training_cfg = enrollment.get("metadata", {}).get("training_config", {})
    if args.preprocess_mode == "auto":
        args.preprocess_mode = training_cfg.get("preprocess_mode", "arcface")
    if args.align_face is None:
        args.align_face = bool(training_cfg.get("align_face", True))
    if args.expected_embedding_dim <= 0:
        args.expected_embedding_dim = int(enrollment.get("metadata", {}).get("embedding_dim", 0))

    workers = max(1, args.workers)
    worker_chunksize = max(1, args.worker_chunksize)
    batch_size = max(1, args.batch_size)
    progress_interval = max(1, args.progress_interval)

    aug_splits = {s.strip().lower() for s in args.aug_splits.split(",") if s.strip()}
    entries = gather_entries(
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
        if status == "embed_dim_mismatch":
            per_bucket[bucket_name].skipped_embed_dim_mismatch += 1
            overall.skipped_embed_dim_mismatch += 1
            return

        truth = result["truth"]
        predicted = result["predicted"]
        used_fallback = bool(result["used_fallback"])

        per_bucket[bucket_name].evaluated_images += 1
        overall.evaluated_images += 1
        if used_fallback:
            per_bucket[bucket_name].face_fallback_used += 1
            overall.face_fallback_used += 1
        else:
            per_bucket[bucket_name].face_detected += 1
            overall.face_detected += 1

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
    processed = 0

    if workers == 1:
        _init_eval_worker(
            args.yunet_model,
            args.mobilefacenet_model,
            args.det_score_threshold,
            args.det_nms_threshold,
            args.det_top_k,
            centroids_payload,
            args.threshold,
            args.assume_processed_are_cropped,
            args.raw_fallback_full_image,
            args.raw_detect_max_side,
            args.raw_clahe,
            args.raw_gamma,
            args.embed_input_size,
            args.preprocess_mode,
            args.align_face,
            args.expected_embedding_dim,
        )

        for idx, sample in enumerate(samples, start=1):
            result = _process_eval_sample(sample)
            aggregate_result(result)
            processed = idx
            if idx == 1 or idx % progress_interval == 0 or idx == total_planned:
                print_progress(idx, total_planned, start, f"{sample.bucket}/{sample.person}")
    else:
        batches = make_batches(samples, batch_size)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_eval_worker,
            initargs=(
                args.yunet_model,
                args.mobilefacenet_model,
                args.det_score_threshold,
                args.det_nms_threshold,
                args.det_top_k,
                centroids_payload,
                args.threshold,
                args.assume_processed_are_cropped,
                args.raw_fallback_full_image,
                args.raw_detect_max_side,
                args.raw_clahe,
                args.raw_gamma,
                args.embed_input_size,
                args.preprocess_mode,
                args.align_face,
                args.expected_embedding_dim,
            ),
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
                    processed = idx
                    if idx == 1 or idx % progress_interval == 0 or idx == total_planned:
                        print_progress(idx, total_planned, start, f"{result['bucket']}/{result['truth']}")

    elapsed = time.time() - start
    if total_planned > 0:
        print_progress(min(processed, total_planned), total_planned, start, "done")
        print()

    thresholds = [
        float(x.strip()) for x in args.threshold_sweep.split(",") if x.strip()
    ]
    threshold_sweep = compute_threshold_sweep(eval_records, thresholds) if thresholds else []

    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)

    report = {
        "model_family": "yunet_mobilefacenet",
        "enrollment_path": args.enrollment_path,
        "threshold": args.threshold,
        "config": {
            "det_score_threshold": args.det_score_threshold,
            "det_nms_threshold": args.det_nms_threshold,
            "det_top_k": args.det_top_k,
            "assume_processed_are_cropped": args.assume_processed_are_cropped,
            "raw_fallback_full_image": args.raw_fallback_full_image,
            "raw_detect_max_side": args.raw_detect_max_side,
            "raw_clahe": args.raw_clahe,
            "raw_gamma": args.raw_gamma,
            "preprocess_mode": args.preprocess_mode,
            "align_face": args.align_face,
            "expected_embedding_dim": args.expected_embedding_dim,
            "progress_interval": args.progress_interval,
            "workers": workers,
            "batch_size": batch_size,
            "worker_chunksize": worker_chunksize,
            "threshold_sweep": args.threshold_sweep,
        },
        "elapsed_seconds": elapsed,
        "buckets": [bucket_to_dict(name, per_bucket[name]) for name in sorted(per_bucket.keys())],
        "overall": bucket_to_dict("overall", overall),
        "threshold_sweep": threshold_sweep,
        "sample_misclassifications": misclassified,
    }

    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[RESULT] YuNet + MobileFaceNet Evaluation")
    print(f"[INFO] Enrollment: {args.enrollment_path}")
    print(f"[INFO] Threshold: {args.threshold:.3f}")
    print(f"[INFO] Evaluated in: {elapsed:.2f}s")
    print("\n[HIT RATE BY DATA BUCKET]")
    for name in sorted(per_bucket.keys()):
        print(summarize_bucket(name, per_bucket[name]))
    print("\n[OVERALL]")
    print(summarize_bucket("overall", overall))
    if overall.evaluated_images > 0:
        print(f"Overall hit rate: {100.0 * overall.correct / overall.evaluated_images:.2f}%")
    print(f"\n[OK] Wrote JSON report to: {args.report_json}")


if __name__ == "__main__":
    main()
