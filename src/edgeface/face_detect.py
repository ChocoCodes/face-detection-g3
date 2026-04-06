import argparse
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.edgeface.common import (
    cosine,
    l2_normalize,
    preprocess_for_edgeface,
    resolve_path,
    root_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live webcam recognition using YuNet detector + EdgeFace embeddings."
    )
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
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--det-score-threshold", type=float, default=0.6)
    parser.add_argument("--det-nms-threshold", type=float, default=0.3)
    parser.add_argument("--det-top-k", type=int, default=5000)
    parser.add_argument("--embed-input-size", type=int, default=0)
    parser.add_argument("--detect-every", type=int, default=2)
    parser.add_argument("--largest-face-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.yunet_model = resolve_path(args.yunet_model)
    args.edgeface_model = resolve_path(args.edgeface_model)
    args.enrollment_path = resolve_path(args.enrollment_path)

    if not Path(args.yunet_model).exists():
        raise FileNotFoundError(f"YuNet model not found: {args.yunet_model}")
    if not Path(args.edgeface_model).exists():
        raise FileNotFoundError(f"EdgeFace model not found: {args.edgeface_model}")
    if not Path(args.enrollment_path).exists():
        raise FileNotFoundError(f"Enrollment file not found: {args.enrollment_path}")

    with open(args.enrollment_path, "r", encoding="utf-8") as f:
        enrollment = json.load(f)

    if args.threshold < 0:
        args.threshold = float(enrollment.get("metadata", {}).get("recommended_threshold", 0.55))

    training_cfg = enrollment.get("metadata", {}).get("training_config", {})
    if args.embed_input_size <= 0:
        args.embed_input_size = int(training_cfg.get("embed_input_size", 112))

    centroids = {
        person: l2_normalize(np.array(payload["centroid"], dtype=np.float32))
        for person, payload in enrollment.get("people", {}).items()
    }
    if not centroids:
        raise RuntimeError("Enrollment JSON has no people data.")

    detector = cv.FaceDetectorYN.create(
        args.yunet_model,
        "",
        (320, 320),
        args.det_score_threshold,
        args.det_nms_threshold,
        args.det_top_k,
    )
    embedder = cv.dnn.readNetFromONNX(args.edgeface_model)

    webcam = cv.VideoCapture(args.camera_index)
    if not webcam.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera_index}")

    print("[INFO] Press 'd' to exit.")
    frame_count = 0
    prev_frame_time = 0.0
    fps_ema = 0.0
    displayed_fps = 0
    cached_draw_items = []

    while True:
        ok, frame = webcam.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        ratio = args.target_width / float(w)
        resized = cv.resize(frame, (args.target_width, int(h * ratio)))

        now = time.time()
        frame_time = now - prev_frame_time
        fps = 1.0 / frame_time if frame_time > 0 else 0.0
        prev_frame_time = now
        fps_ema = fps if fps_ema == 0.0 else 0.85 * fps_ema + 0.15 * fps

        frame_count += 1
        if frame_count % 5 == 0:
            displayed_fps = int(fps_ema)

        if frame_count % max(1, args.detect_every) == 0 or not cached_draw_items:
            rh, rw = resized.shape[:2]
            detector.setInputSize((rw, rh))
            _, faces = detector.detect(resized)
            cached_draw_items = []
            if faces is not None and len(faces) > 0:
                faces_list = list(faces)
                if args.largest_face_only:
                    faces_list = [max(faces_list, key=lambda row: float(row[2] * row[3]))]

                for face_row in faces_list:
                    x, y, bw, bh = face_row[:4]
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(rw, int(x + bw))
                    y2 = min(rh, int(y + bh))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    face_crop = resized[y1:y2, x1:x2]
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

                    recognized = best_score >= args.threshold
                    label = best_person if recognized else "Unknown"
                    color = (0, 255, 0) if recognized else (0, 0, 255)
                    draw_box = (
                        int(x1 / ratio),
                        int(y1 / ratio),
                        int((x2 - x1) / ratio),
                        int((y2 - y1) / ratio),
                    )
                    cached_draw_items.append((draw_box, label, best_score, color))

        for draw_box, label, score, color in cached_draw_items:
            x, y, bw, bh = draw_box
            cv.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv.putText(
                frame,
                f"{label} ({score:.2f})",
                (x, max(0, y - 10)),
                cv.FONT_HERSHEY_COMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

        cv.putText(
            frame,
            f"FPS: {displayed_fps}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv.imshow("YuNet + EdgeFace Live", frame)

        if cv.waitKey(1) & 0xFF == ord("d"):
            break

    webcam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
