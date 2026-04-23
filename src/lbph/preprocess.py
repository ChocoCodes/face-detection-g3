import math
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

IMG_SIZE = (100, 100)


@dataclass
class FacePreprocessResult:
    face: np.ndarray | None
    reason: str | None
    detected_face: bool
    used_alignment: bool
    face_box: tuple[int, int, int, int] | None


def resolve_eye_cascade_path(configured_path: str | None) -> str:
    if configured_path:
        candidate = Path(configured_path)
        if candidate.exists():
            return str(candidate)
    return str(Path(cv.data.haarcascades) / "haarcascade_eye.xml")


def maybe_downscale(image_gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    if max_side <= 0:
        return image_gray, 1.0
    h, w = image_gray.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_gray, 1.0
    scale = max_side / float(longest)
    resized = cv.resize(image_gray, (int(w * scale), int(h * scale)))
    return resized, scale


def detect_largest_face_box(
    image_gray: np.ndarray,
    face_cascade: cv.CascadeClassifier,
    min_face_size: int,
    scale_factor: float,
    min_neighbors: int,
) -> tuple[int, int, int, int] | None:
    faces = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    return int(x), int(y), int(w), int(h)


def align_face_by_eyes(
    face_gray: np.ndarray,
    eye_cascade: cv.CascadeClassifier,
) -> tuple[np.ndarray, bool]:
    h, w = face_gray.shape[:2]
    if h < 20 or w < 20:
        return face_gray, False

    upper = face_gray[: max(1, int(h * 0.65)), :]
    eyes = eye_cascade.detectMultiScale(
        upper,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(max(8, w // 12), max(8, h // 12)),
    )
    if len(eyes) < 2:
        return face_gray, False

    eye_boxes = sorted(eyes, key=lambda e: int(e[2]) * int(e[3]), reverse=True)[:4]
    centers: list[tuple[float, float]] = []
    for ex, ey, ew, eh in eye_boxes:
        centers.append((float(ex + ew / 2.0), float(ey + eh / 2.0)))

    best_pair: tuple[tuple[float, float], tuple[float, float]] | None = None
    best_dist = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            c1 = centers[i]
            c2 = centers[j]
            if abs(c1[1] - c2[1]) > (h * 0.25):
                continue
            dist = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
            if dist > best_dist:
                best_dist = dist
                best_pair = (c1, c2)

    if best_pair is None or best_dist <= 1.0:
        return face_gray, False

    left_eye, right_eye = sorted(best_pair, key=lambda c: c[0])
    angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    center = (w / 2.0, h / 2.0)
    rot = cv.getRotationMatrix2D(center, -angle, 1.0)
    aligned = cv.warpAffine(
        face_gray,
        rot,
        (w, h),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE,
    )
    return aligned, True


def normalize_face(
    face_gray: np.ndarray,
    img_size: tuple[int, int],
    equalization: str,
) -> np.ndarray:
    if equalization == "clahe":
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(face_gray)
    else:
        normalized = cv.equalizeHist(face_gray)
    return cv.resize(normalized, img_size)


def extract_lbph_face(
    image_gray: np.ndarray,
    face_cascade: cv.CascadeClassifier,
    min_face_size: int,
    scale_factor: float,
    min_neighbors: int,
    img_size: tuple[int, int] = IMG_SIZE,
    equalization: str = "equalize",
    align_eyes: bool = True,
    eye_cascade: cv.CascadeClassifier | None = None,
    downscale_max_side: int = 0,
) -> FacePreprocessResult:
    h, w = image_gray.shape[:2]
    if h < min_face_size or w < min_face_size:
        return FacePreprocessResult(
            face=None,
            reason="image_too_small",
            detected_face=False,
            used_alignment=False,
            face_box=None,
        )

    detect_gray, scale = maybe_downscale(image_gray, downscale_max_side)
    face_box = detect_largest_face_box(
        image_gray=detect_gray,
        face_cascade=face_cascade,
        min_face_size=min_face_size,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
    )
    if face_box is None:
        return FacePreprocessResult(
            face=None,
            reason="no_face",
            detected_face=False,
            used_alignment=False,
            face_box=None,
        )

    x, y, fw, fh = face_box
    if scale != 1.0:
        x = int(round(x / scale))
        y = int(round(y / scale))
        fw = int(round(fw / scale))
        fh = int(round(fh / scale))

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    fw = max(1, min(fw, w - x))
    fh = max(1, min(fh, h - y))

    face_roi = image_gray[y : y + fh, x : x + fw]
    if face_roi.size == 0:
        return FacePreprocessResult(
            face=None,
            reason="no_face",
            detected_face=False,
            used_alignment=False,
            face_box=None,
        )

    aligned = False
    if align_eyes and eye_cascade is not None and not eye_cascade.empty():
        face_roi, aligned = align_face_by_eyes(face_roi, eye_cascade)

    normalized = normalize_face(face_roi, img_size=img_size, equalization=equalization)
    return FacePreprocessResult(
        face=normalized,
        reason=None,
        detected_face=True,
        used_alignment=aligned,
        face_box=(x, y, fw, fh),
    )
