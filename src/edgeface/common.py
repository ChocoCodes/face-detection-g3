import os
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


def is_image_file(file_name: str) -> bool:
    _, ext = os.path.splitext(file_name)
    return ext.lower() in ALLOWED_EXTENSIONS


@dataclass
class Sample:
    bucket: str
    person: str
    path: str


def gather_samples(
    base_data_dir: str,
    raw_dir: str,
    processed_dir: str,
    augmented_dir: str,
    aug_splits: set[str],
    include_raw: bool,
    include_processed: bool,
    include_augmented: bool,
    max_images_per_person: int,
) -> list[Sample]:
    samples: list[Sample] = []
    used: dict[tuple[str, str], int] = {}

    def add_bucket(bucket_name: str, person_name: str, person_path: str) -> None:
        files = [f for f in sorted(os.listdir(person_path)) if is_image_file(f)]
        for image_name in files:
            key = (bucket_name, person_name)
            count = used.get(key, 0)
            if max_images_per_person > 0 and count >= max_images_per_person:
                break
            used[key] = count + 1
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

    if include_augmented:
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


def preprocess_for_edgeface(face_bgr: np.ndarray, input_size: int) -> np.ndarray:
    face = cv.resize(face_bgr, (input_size, input_size), interpolation=cv.INTER_AREA)
    face_rgb = cv.cvtColor(face, cv.COLOR_BGR2RGB).astype(np.float32)
    face_rgb = (face_rgb - 127.5) / 128.0
    chw = np.transpose(face_rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


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


def extract_face_crop(
    img_bgr: np.ndarray,
    detector: cv.FaceDetectorYN,
    detect_max_side: int,
    input_size: int,
    align_face: bool,
) -> tuple[np.ndarray | None, bool]:
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
        return None, False

    best = max(faces, key=lambda row: float(row[14]))
    if align_face and len(best) >= 14:
        landmarks = np.array(best[4:14], dtype=np.float32).reshape(5, 2)
        landmarks /= scale
        aligned = align_face_by_landmarks(img_bgr, landmarks, input_size)
        if aligned is not None:
            return aligned, True

    x, y, bw, bh = best[:4]
    x1 = int(max(0, x / scale))
    y1 = int(max(0, y / scale))
    x2 = int(min(w, (x + bw) / scale))
    y2 = int(min(h, (y + bh) / scale))
    if x2 <= x1 or y2 <= y1:
        return None, False
    return img_bgr[y1:y2, x1:x2], True


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return v
    return v / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
