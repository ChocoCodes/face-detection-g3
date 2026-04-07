#!/usr/bin/env python3
"""
Real-time face detection and recognition using ArcFace.

Displays live webcam feed with detected faces, embeddings, and
identified individuals from enrollment, with temporal smoothing.
"""

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np

try:
    import insightface
except ImportError:
    print("[ERROR] insightface not installed. Run: pip install insightface")
    sys.exit(1)

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
        description="Real-time face recognition using ArcFace."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera ID.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps-target", type=int, default=30)
    parser.add_argument(
        "--model-dir",
        default=root_path("models", "arcface_mobilenet"),
        help="Directory containing ArcFace model.",
    )
    parser.add_argument(
        "--enrollment-path",
        default=root_path("models", "arcface_mobilenet", "enrollment.json"),
        help="Path to enrollment JSON.",
    )
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument(
        "--fps-log-path",
        default=root_path("reports", "benchmark", "live_fps", "arcface_fps.jsonl"),
        help="Path to append periodic FPS samples as JSONL.",
    )
    parser.add_argument(
        "--fps-log-interval",
        type=float,
        default=1.0,
        help="Seconds between FPS log writes.",
    )
    parser.add_argument(
        "--fps-summary-dir",
        default=root_path("reports", "benchmark", "live_fps", "runs"),
        help="Directory to write per-run FPS summary JSON.",
    )
    parser.add_argument("--disable-fps-log", action="store_true")
    return parser.parse_args()


def load_enrollment(path: str) -> tuple[dict, dict]:
    """Load centroids and metadata from enrollment JSON."""
    with open(path, "r", encoding="utf-8") as f:
        enrollment = json.load(f)
    
    people = sorted(enrollment["people"].keys())
    centroids = {
        person: np.array(enrollment["people"][person]["centroid"], dtype=np.float32)
        for person in people
    }
    
    metadata = enrollment["metadata"]
    
    return centroids, metadata


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding to unit length."""
    norm = np.linalg.norm(embedding)
    if norm > 1e-12:
        return embedding / norm
    return embedding


def match_face(embedding: np.ndarray, centroids: dict, threshold: float) -> tuple[str, float]:
    """Match face embedding against enrolled centroids."""
    best_person = "Unknown"
    best_score = -1.0
    
    for person, centroid in centroids.items():
        score = float(np.dot(embedding, centroid))
        if score > best_score:
            best_score = score
            best_person = person
    
    if best_score < threshold:
        best_person = "Unknown"
    
    return best_person, best_score


def main() -> None:
    args = parse_args()
    args.model_dir = resolve_path(args.model_dir)
    args.enrollment_path = resolve_path(args.enrollment_path)
    args.fps_log_path = resolve_path(args.fps_log_path)
    args.fps_summary_dir = resolve_path(args.fps_summary_dir)
    
    if not os.path.exists(args.enrollment_path):
        print(f"[ERROR] Enrollment not found: {args.enrollment_path}")
        print(f"[HINT] Run: python src/arcface_mobilenet/trainer.py")
        sys.exit(1)
    
    centroids, metadata = load_enrollment(args.enrollment_path)
    
    if args.threshold < 0:
        args.threshold = float(metadata.get("recommended_threshold", 0.55))
    
    print(f"[INFO] Loaded {len(centroids)} enrolled identities")
    print(f"[INFO] Threshold: {args.threshold:.3f}")
    print(f"[INFO] Opening camera {args.camera}...")
    
    try:
        aface = insightface.app.FaceAnalysis(
            name="buffalo_s",
            root=args.model_dir,
            providers=["CPUExecutionProvider"],
        )
        aface.prepare(ctx_id=-1, det_thresh=args.det_thresh, det_size=(args.det_size, args.det_size))
    except Exception as e:
        print(f"[ERROR] Failed to load ArcFace: {e}")
        sys.exit(1)
    
    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera {args.camera}")
        sys.exit(1)
    
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv.CAP_PROP_FPS, args.fps_target)
    
    smooth_window = max(1, args.smooth_window)
    detections_history = deque(maxlen=smooth_window)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fps_log_interval = max(0.2, args.fps_log_interval)
    fps_log_file = None
    if not args.disable_fps_log:
        fps_log_path = Path(args.fps_log_path)
        fps_log_path.parent.mkdir(parents=True, exist_ok=True)
        fps_log_file = fps_log_path.open("a", encoding="utf-8")

    run_started = cv.getTickCount()
    last_log_time = 0.0
    fps_sum = 0.0
    fps_samples = 0
    
    frame_count = 0
    start_frame_time = None
    last_fps = 0.0
    
    print("\n[READY] Press 'q' to exit, 's' to save frame.")
    print("-" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        frame_count += 1
        if start_frame_time is None:
            start_frame_time = cv.getTickCount()
        
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        try:
            faces = aface.get(rgb_frame)
        except Exception:
            faces = []
        
        current_detections = []
        for face in faces:
            x1, y1, x2, y2 = [int(c) for c in face.bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            embedding = normalize_embedding(face.embedding.astype(np.float32))
            person, score = match_face(embedding, centroids, args.threshold)
            
            current_detections.append(
                {
                    "rect": (x1, y1, x2, y2),
                    "person": person,
                    "score": score,
                    "det_score": face.det_score,
                }
            )
        
        detections_history.append(current_detections)
        
        smoothed = []
        for i in range(len(current_detections)):
            votes = defaultdict(int)
            for hist_detections in detections_history:
                if i < len(hist_detections):
                    votes[hist_detections[i]["person"]] += 1
            if votes:
                person = max(votes, key=votes.get)
                score = current_detections[i]["score"] if i < len(current_detections) else 0.0
                smoothed.append(
                    {
                        "rect": current_detections[i]["rect"],
                        "person": person,
                        "score": score,
                    }
                )
        
        for detection in smoothed:
            x1, y1, x2, y2 = detection["rect"]
            person = detection["person"]
            score = detection["score"]
            
            if person == "Unknown":
                color = (0, 0, 255)
                label_text = f"Unknown ({score:.2f})"
            else:
                color = (0, 255, 0)
                label_text = f"{person} ({score:.2f})"
            
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_size = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y = max(20, y1 - 10)
            label_x = max(0, x1)
            cv.rectangle(
                frame,
                (label_x, label_y - label_size[1] - 5),
                (label_x + label_size[0], label_y + 5),
                color,
                -1,
            )
            cv.putText(
                frame,
                label_text,
                (label_x, label_y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        
        elapsed_ticks = cv.getTickCount() - start_frame_time
        elapsed_seconds = elapsed_ticks / cv.getTickFrequency()
        if elapsed_seconds > 0:
            last_fps = frame_count / elapsed_seconds
            fps_sum += last_fps
            fps_samples += 1

            if fps_log_file is not None and (elapsed_seconds - last_log_time) >= fps_log_interval:
                sample = {
                    "run_id": run_id,
                    "algorithm": "arcface",
                    "elapsed_seconds": elapsed_seconds,
                    "frame_count": frame_count,
                    "fps": last_fps,
                }
                fps_log_file.write(json.dumps(sample) + "\n")
                fps_log_file.flush()
                last_log_time = elapsed_seconds
        
        info_text = f"FPS: {last_fps:.1f} | Faces: {len(smoothed)} | Frames: {frame_count}"
        cv.putText(
            frame,
            info_text,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        
        cv.imshow("ArcFace Face Recognition", frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[INFO] Exiting...")
            break
        elif key == ord("s"):
            filename = f"arcface_frame_{frame_count:06d}.png"
            filepath = os.path.join(PROJECT_ROOT, filename)
            cv.imwrite(filepath, frame)
            print(f"[OK] Saved: {filepath}")
    
    cap.release()
    cv.destroyAllWindows()
    total_seconds = (cv.getTickCount() - run_started) / cv.getTickFrequency()
    avg_fps = (fps_sum / fps_samples) if fps_samples else 0.0

    Path(args.fps_summary_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.fps_summary_dir) / f"arcface_{run_id}.json"
    summary_payload = {
        "run_id": run_id,
        "algorithm": "arcface",
        "average_fps": avg_fps,
        "frames": frame_count,
        "duration_seconds": total_seconds,
        "threshold": args.threshold,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    if fps_log_file is not None:
        fps_log_file.close()

    print(f"[FPS] Average FPS: {avg_fps:.2f}")
    print(f"[FPS] Summary: {summary_path}")
    print(f"[DONE] Processed {frame_count} frames")


if __name__ == "__main__":
    from collections import defaultdict
    main()
