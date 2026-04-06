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
        description="Build identity enrollment using YuNet detection + MobileFaceNet embeddings."
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
        help="Include raw dataset during enrollment build.",
    )
    parser.add_argument(
        "--max-images-per-person",
        type=int,
        default=0,
        help="Optional cap per person per bucket (0 = no cap).",
    )
    parser.add_argument(
        "--yunet-model",
        default=root_path("models", "yunet_mobilefacenet", "face_detection_yunet_2023mar.onnx"),
    )
    parser.add_argument(
        "--mobilefacenet-model",
        default=root_path("models", "yunet_mobilefacenet", "mobilefacenet.onnx"),
    )
    parser.add_argument(
        "--enrollment-output",
        default=root_path("models", "yunet_mobilefacenet", "enrollment.json"),
    )
    parser.add_argument("--det-score-threshold", type=float, default=0.6)
    parser.add_argument("--det-nms-threshold", type=float, default=0.3)
    parser.add_argument("--det-top-k", type=int, default=5000)
    parser.add_argument("--embed-input-size", type=int, default=112)
    parser.add_argument(
        "--preprocess-mode",
        choices=["arcface", "mobilefacenet"],
        default="arcface",
        help="Input normalization contract for embedding model.",
    )
    parser.add_argument(
        "--align-face",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use YuNet landmarks for affine face alignment before embedding.",
    )
    parser.add_argument(
        "--expected-embedding-dim",
        type=int,
        default=0,
        help="Optional embedding dim contract check (0 disables).",
    )
    parser.add_argument(
        "--assume-processed-are-cropped",
        action="store_true",
        help="Skip detection for processed/augmented buckets and use full frame as face ROI.",
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
        "--raw-weight",
        type=float,
        default=2.0,
        help="Relative weight of raw samples when computing identity centroids.",
    )
    parser.add_argument(
        "--threshold-candidates",
        default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75",
        help="Cosine thresholds to evaluate for a suggested operating point.",
    )
    parser.add_argument(
        "--calibration-objective",
        choices=["balanced_accuracy", "id_accuracy"],
        default="balanced_accuracy",
        help="Objective used to select recommended threshold.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for embedding extraction (1 = sequential).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Progress print interval in number of processed samples.",
    )
    parser.add_argument(
        "--worker-chunksize",
        type=int,
        default=64,
        help="ProcessPool map chunksize; larger values usually reduce IPC overhead.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Samples per worker task in multiprocessing mode.",
    )
    return parser.parse_args()


def is_image_file(file_name: str) -> bool:
    _, ext = os.path.splitext(file_name)
    return ext.lower() in ALLOWED_EXTENSIONS


@dataclass
class Sample:
    bucket: str
    person: str
    path: str


_WORKER_DETECTOR = None
_WORKER_EMBEDDER = None
_WORKER_ASSUME_PROCESSED = False
_WORKER_RAW_FALLBACK_FULL_IMAGE = True
_WORKER_RAW_DETECT_MAX_SIDE = 640
_WORKER_RAW_CLAHE = True
_WORKER_RAW_GAMMA = 1.15
_WORKER_EMBED_INPUT_SIZE = 112
_WORKER_PREPROCESS_MODE = "arcface"
_WORKER_ALIGN_FACE = True


def _init_worker(
    yunet_model: str,
    mobilefacenet_model: str,
    det_score_threshold: float,
    det_nms_threshold: float,
    det_top_k: int,
    assume_processed_are_cropped: bool,
    raw_fallback_full_image: bool,
    raw_detect_max_side: int,
    raw_clahe: bool,
    raw_gamma: float,
    embed_input_size: int,
    preprocess_mode: str,
    align_face: bool,
) -> None:
    global _WORKER_DETECTOR
    global _WORKER_EMBEDDER
    global _WORKER_ASSUME_PROCESSED
    global _WORKER_RAW_FALLBACK_FULL_IMAGE
    global _WORKER_RAW_DETECT_MAX_SIDE
    global _WORKER_RAW_CLAHE
    global _WORKER_RAW_GAMMA
    global _WORKER_EMBED_INPUT_SIZE
    global _WORKER_PREPROCESS_MODE
    global _WORKER_ALIGN_FACE

    cv.setNumThreads(1)
    _WORKER_DETECTOR = cv.FaceDetectorYN.create(
        yunet_model,
        "",
        (320, 320),
        det_score_threshold,
        det_nms_threshold,
        det_top_k,
    )
    _WORKER_EMBEDDER = cv.dnn.readNetFromONNX(mobilefacenet_model)
    _WORKER_ASSUME_PROCESSED = assume_processed_are_cropped
    _WORKER_RAW_FALLBACK_FULL_IMAGE = raw_fallback_full_image
    _WORKER_RAW_DETECT_MAX_SIDE = raw_detect_max_side
    _WORKER_RAW_CLAHE = raw_clahe
    _WORKER_RAW_GAMMA = raw_gamma
    _WORKER_EMBED_INPUT_SIZE = embed_input_size
    _WORKER_PREPROCESS_MODE = preprocess_mode
    _WORKER_ALIGN_FACE = align_face


def _extract_embedding(
    sample: Sample,
    detector: cv.FaceDetectorYN,
    embedder,
    assume_processed_are_cropped: bool,
    raw_fallback_full_image: bool,
    raw_detect_max_side: int,
    raw_clahe: bool,
    raw_gamma: float,
    embed_input_size: int,
    preprocess_mode: str,
    align_face: bool,
):
    img = cv.imread(sample.path)
    if img is None:
        return "unreadable", None

    if assume_processed_are_cropped and sample.bucket != "raw":
        face_crop = img
    else:
        detect_max_side = raw_detect_max_side if sample.bucket == "raw" else 0
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
        detector.setInputSize((dw, dh))
        _, faces = detector.detect(detect_img)

        aligned_face = None
        face_crop = None
        if faces is not None and len(faces) > 0:
            best = max(faces, key=lambda row: float(row[14]))
            if align_face and len(best) >= 14:
                landmarks = np.array(best[4:14], dtype=np.float32).reshape(5, 2)
                landmarks /= scale
                aligned_face = align_face_by_landmarks(img, landmarks, embed_input_size)

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
            if sample.bucket == "raw" and raw_fallback_full_image:
                face_crop = img
            else:
                return "no_face", None

    if sample.bucket == "raw":
        face_crop = enhance_raw_face(face_crop, raw_clahe, raw_gamma)

    blob = preprocess_for_mobilefacenet(face_crop, embed_input_size, preprocess_mode)
    embedder.setInput(blob)
    emb = embedder.forward().flatten().astype(np.float32)
    emb = l2_normalize(emb)
    return "ok", emb


def _process_sample_worker(sample: Sample):
    status, emb = _extract_embedding(
        sample=sample,
        detector=_WORKER_DETECTOR,
        embedder=_WORKER_EMBEDDER,
        assume_processed_are_cropped=_WORKER_ASSUME_PROCESSED,
        raw_fallback_full_image=_WORKER_RAW_FALLBACK_FULL_IMAGE,
        raw_detect_max_side=_WORKER_RAW_DETECT_MAX_SIDE,
        raw_clahe=_WORKER_RAW_CLAHE,
        raw_gamma=_WORKER_RAW_GAMMA,
        embed_input_size=_WORKER_EMBED_INPUT_SIZE,
        preprocess_mode=_WORKER_PREPROCESS_MODE,
        align_face=_WORKER_ALIGN_FACE,
    )
    if emb is None:
        return status, sample.person, sample.bucket, None
    return status, sample.person, sample.bucket, emb.tolist()


def _process_batch_worker(batch_samples: List[Sample]):
    out = []
    for sample in batch_samples:
        status, emb = _extract_embedding(
            sample=sample,
            detector=_WORKER_DETECTOR,
            embedder=_WORKER_EMBEDDER,
            assume_processed_are_cropped=_WORKER_ASSUME_PROCESSED,
            raw_fallback_full_image=_WORKER_RAW_FALLBACK_FULL_IMAGE,
            raw_detect_max_side=_WORKER_RAW_DETECT_MAX_SIDE,
            raw_clahe=_WORKER_RAW_CLAHE,
            raw_gamma=_WORKER_RAW_GAMMA,
            embed_input_size=_WORKER_EMBED_INPUT_SIZE,
            preprocess_mode=_WORKER_PREPROCESS_MODE,
            align_face=_WORKER_ALIGN_FACE,
        )
        if emb is None:
            out.append((status, sample.person, sample.bucket, None))
        else:
            out.append((status, sample.person, sample.bucket, emb.tolist()))
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
    max_images_per_person: int,
) -> List[Sample]:
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
        f"| {rate:6.2f} img/s | {status:<20}"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()


def main() -> None:
    args = parse_args()
    args.base_data_dir = resolve_path(args.base_data_dir)
    args.yunet_model = resolve_path(args.yunet_model)
    args.mobilefacenet_model = resolve_path(args.mobilefacenet_model)
    args.enrollment_output = resolve_path(args.enrollment_output)

    if not os.path.exists(args.yunet_model):
        raise FileNotFoundError(
            f"YuNet model not found: {args.yunet_model}. Place ONNX model at this path or pass --yunet-model."
        )
    if not os.path.exists(args.mobilefacenet_model):
        raise FileNotFoundError(
            f"MobileFaceNet model not found: {args.mobilefacenet_model}. Place ONNX model at this path or pass --mobilefacenet-model."
        )

    os.makedirs(os.path.dirname(args.enrollment_output), exist_ok=True)

    workers = max(1, args.workers)
    progress_interval = max(1, args.progress_interval)
    worker_chunksize = max(1, args.worker_chunksize)
    batch_size = max(1, args.batch_size)

    detector = None
    embedder = None
    if workers == 1:
        detector = cv.FaceDetectorYN.create(
            args.yunet_model,
            "",
            (320, 320),
            args.det_score_threshold,
            args.det_nms_threshold,
            args.det_top_k,
        )
        embedder = cv.dnn.readNetFromONNX(args.mobilefacenet_model)

    aug_splits = {s.strip().lower() for s in args.aug_splits.split(",") if s.strip()}
    samples = gather_samples(
        base_data_dir=args.base_data_dir,
        raw_dir=args.raw_dir_name,
        processed_dir=args.processed_dir_name,
        augmented_dir=args.augmented_dir_name,
        aug_splits=aug_splits,
        include_raw=args.include_raw,
        max_images_per_person=args.max_images_per_person,
    )
    if not samples:
        raise RuntimeError("No training samples found.")

    per_person_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    per_person_weights: Dict[str, List[float]] = defaultdict(list)
    diagnostics = {
        "total_samples": len(samples),
        "used_samples": 0,
        "skipped_unreadable": 0,
        "skipped_no_face": 0,
        "bucket_counts": defaultdict(int),
    }

    start = time.time()
    status_name = "embedding extraction"
    if workers == 1:
        for idx, sample in enumerate(samples, start=1):
            status, emb = _extract_embedding(
                sample=sample,
                detector=detector,
                embedder=embedder,
                assume_processed_are_cropped=args.assume_processed_are_cropped,
                raw_fallback_full_image=args.raw_fallback_full_image,
                raw_detect_max_side=args.raw_detect_max_side,
                raw_clahe=args.raw_clahe,
                raw_gamma=args.raw_gamma,
                embed_input_size=args.embed_input_size,
                preprocess_mode=args.preprocess_mode,
                align_face=args.align_face,
            )

            if status == "unreadable":
                diagnostics["skipped_unreadable"] += 1
            elif status == "no_face":
                diagnostics["skipped_no_face"] += 1
            else:
                if args.expected_embedding_dim > 0 and emb.shape[0] != args.expected_embedding_dim:
                    raise RuntimeError(
                        "Embedding dimension mismatch. "
                        f"Expected {args.expected_embedding_dim}, got {emb.shape[0]} from {sample.path}."
                    )
                per_person_embeddings[sample.person].append(emb)
                sample_weight = args.raw_weight if sample.bucket == "raw" else 1.0
                per_person_weights[sample.person].append(sample_weight)
                diagnostics["used_samples"] += 1
                diagnostics["bucket_counts"][sample.bucket] += 1

            if idx % progress_interval == 0 or idx == len(samples):
                print_progress(idx, len(samples), start, status_name)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                args.yunet_model,
                args.mobilefacenet_model,
                args.det_score_threshold,
                args.det_nms_threshold,
                args.det_top_k,
                args.assume_processed_are_cropped,
                args.raw_fallback_full_image,
                args.raw_detect_max_side,
                args.raw_clahe,
                args.raw_gamma,
                args.embed_input_size,
                args.preprocess_mode,
                args.align_face,
            ),
        ) as executor:
            batches = make_batches(samples, batch_size)
            batch_results = executor.map(
                _process_batch_worker,
                batches,
                chunksize=worker_chunksize,
            )
            idx = 0
            for batch in batch_results:
                for result in batch:
                    idx += 1
                    status, person, bucket, emb_list = result

                    if status == "unreadable":
                        diagnostics["skipped_unreadable"] += 1
                    elif status == "no_face":
                        diagnostics["skipped_no_face"] += 1
                    else:
                        emb = np.array(emb_list, dtype=np.float32)
                        if args.expected_embedding_dim > 0 and emb.shape[0] != args.expected_embedding_dim:
                            raise RuntimeError(
                                "Embedding dimension mismatch. "
                                f"Expected {args.expected_embedding_dim}, got {emb.shape[0]} in worker result."
                            )
                        per_person_embeddings[person].append(emb)
                        sample_weight = args.raw_weight if bucket == "raw" else 1.0
                        per_person_weights[person].append(sample_weight)
                        diagnostics["used_samples"] += 1
                        diagnostics["bucket_counts"][bucket] += 1

                    if idx % progress_interval == 0 or idx == len(samples):
                        print_progress(idx, len(samples), start, status_name)

    if len(samples) > 0:
        print_progress(len(samples), len(samples), start, "done")
        print()

    if not per_person_embeddings:
        raise RuntimeError("No embeddings extracted. Check model files and detection settings.")

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
                "known_reject_rate": 1.0 - tpr,
                "unknown_reject_rate": tnr,
                "unknown_accept_rate": 1.0 - tnr,
                "balanced_accuracy": 0.5 * (tpr + tnr),
            }
        )

    best_threshold = max(
        threshold_scores,
        key=lambda x: select_threshold_score(x, args.calibration_objective),
    )["threshold"]

    enrollment = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "yunet_model": args.yunet_model,
            "mobilefacenet_model": args.mobilefacenet_model,
            "embedding_dim": int(next(iter(centroids.values())).shape[0]),
            "recommended_threshold": float(best_threshold),
            "calibration_objective": args.calibration_objective,
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
                "aug_splits": sorted(aug_splits),
                "assume_processed_are_cropped": args.assume_processed_are_cropped,
                "raw_fallback_full_image": args.raw_fallback_full_image,
                "raw_detect_max_side": args.raw_detect_max_side,
                "raw_clahe": args.raw_clahe,
                "raw_gamma": args.raw_gamma,
                "raw_weight": args.raw_weight,
                "max_images_per_person": args.max_images_per_person,
                "embed_input_size": args.embed_input_size,
                "preprocess_mode": args.preprocess_mode,
                "align_face": args.align_face,
                "expected_embedding_dim": args.expected_embedding_dim,
                "calibration_objective": args.calibration_objective,
                "workers": workers,
                "batch_size": batch_size,
                "worker_chunksize": worker_chunksize,
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
    print(f"[OK] Wrote enrollment: {args.enrollment_output}")
    print(f"[INFO] People enrolled: {len(centroids)}")
    print(f"[INFO] Used samples: {diagnostics['used_samples']} / {diagnostics['total_samples']}")
    print(f"[INFO] Recommended threshold: {best_threshold:.2f}")
    print(f"[TIME] Total time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
