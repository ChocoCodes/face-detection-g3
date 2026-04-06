# Compact Face Models (Beyond Viola-Jones + LBPH)

This note lists strong, lightweight alternatives for this project.

Scope:
- Face detection (replace Haar cascade / Viola-Jones)
- Face recognition (replace LBPH)
- Focus on compact + fast + accurate models suitable for laptop/webcam use

## Quick Answer (Recommended)

If you want the best compact upgrade with strong accuracy:
1. Detector: YuNet (OpenCV) or SCRFD-500M (ONNX Runtime)
2. Recognizer: MobileFaceNet with ArcFace embeddings
3. Matching: cosine similarity against enrolled embeddings

Why this is the best balance:
- Much better robustness than Haar/LBPH for pose, lighting, and partial occlusion.
- Still lightweight enough for real-time CPU inference.
- Easy to scale from 28 classes to more identities without retraining classifier heads.

## Detection Models (Compact)

### 1) YuNet (Tiny, OpenCV-friendly)
- Type: CNN face detector
- Strengths:
  - Designed to be tiny and millisecond-level.
  - Directly available in OpenCV Zoo ONNX models.
  - Easy drop-in for OpenCV pipelines.
- Tradeoff:
  - Usually not the top performer on hardest tiny-face cases, but excellent size/speed/accuracy balance.
- Best use:
  - Webcam attendance/recognition at close-to-medium distance.

### 2) BlazeFace (Ultra-fast mobile detector)
- Type: lightweight mobile GPU detector
- Strengths:
  - Extremely fast; reported 200-1000+ FPS on flagship mobile devices.
  - Great for real-time front-camera scenarios.
- Tradeoff:
  - Ecosystem is stronger in MediaPipe/mobile stacks than plain OpenCV-only stacks.
- Best use:
  - If you want very low latency and can use MediaPipe/TFLite pipelines.

### 3) SCRFD-500M / 1G (Best accuracy-per-compute in compact class)
- Type: efficient anchor-based detector family
- Strengths:
  - Excellent efficiency-accuracy trade-off.
  - Lightweight variants (500M, 1G) are compact and practical on CPU.
  - Stronger hard-case robustness than many tiny detectors.
- Tradeoff:
  - Integration is usually through ONNX Runtime (a bit more setup than Haar/YuNet).
- Best use:
  - If you want the strongest detector quality while staying lightweight.

### 4) RetinaFace-MobileNet0.25 (Good baseline)
- Type: RetinaFace with tiny MobileNet backbone
- Strengths:
  - Popular baseline; good quality for a small model.
- Tradeoff:
  - Newer compact detectors (SCRFD/YuNet) often provide better practical trade-offs.
- Best use:
  - If you already have RetinaFace tooling.

## Recognition Models (Compact)

### 1) MobileFaceNet + ArcFace embeddings (Top compact choice)
- Type: embedding model (not a direct classifier)
- Strengths:
  - Sub-1M parameter design family; very efficient.
  - Reported around 4MB model size in original paper context.
  - Strong verification/identification quality for mobile/edge.
- Tradeoff:
  - Requires embedding matching pipeline (cosine similarity + threshold), not LBPH predict().
- Best use:
  - Recommended default replacement for LBPH in this project.

### 2) GhostFaceNet (Very small recognition network)
- Type: compact embedding model
- Strengths:
  - Very small footprint; good when memory/CPU are strict.
- Tradeoff:
  - Accuracy can be below larger modern embedding models, depending on variant and training.
- Best use:
  - Extreme edge constraints.

### 3) FaceNet (with lightweight backbone variants)
- Type: embedding model
- Strengths:
  - Mature metric-learning approach.
- Tradeoff:
  - Some deployments are heavier than MobileFaceNet-based stacks.
- Best use:
  - When you already have FaceNet tooling or pretrained assets.

## Practical Ranking For Your Current Project

Given your webcam + local class roster setup:

### Detection ranking (compact first)
1. YuNet (easiest, fastest migration in OpenCV ecosystem)
2. SCRFD-500M (best robustness with still compact footprint)
3. BlazeFace (excellent speed, especially with MediaPipe/TFLite)
4. RetinaFace-MobileNet0.25

### Recognition ranking (compact first)
1. MobileFaceNet + ArcFace embeddings
2. GhostFaceNet
3. FaceNet-light variants

## Suggested Production Stack

### Option A (fastest to integrate in OpenCV world)
- Detector: YuNet
- Recognizer: MobileFaceNet ONNX/TFLite
- Matcher: cosine similarity + class centroids/templates

### Option B (highest compact robustness)
- Detector: SCRFD-500M
- Recognizer: MobileFaceNet
- Matcher: cosine similarity + adaptive threshold per person

## Why Move Away From Haar + LBPH

Haar + LBPH is simple and light, but in practice it is weaker for:
- Non-frontal faces
- Illumination shifts
- Occlusion and expression changes
- Domain transfer when training/evaluation conditions differ

Modern compact CNN detector + embedding pipelines generally improve all of the above while remaining real-time on CPU.

## Migration Notes For This Repo

Short transition plan:
1. Keep your current data folders and label names.
2. Replace face detector step with YuNet (or SCRFD).
3. Replace LBPH training with embedding extraction for each image.
4. Save per-person embedding prototypes (mean vectors) to disk.
5. At runtime, predict identity by nearest cosine distance and threshold.

## References (Research + Official Sources)

- BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs (arXiv:1907.05047)
  - https://arxiv.org/abs/1907.05047
- MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices (arXiv:1804.07573)
  - https://arxiv.org/abs/1804.07573
- SCRFD: Sample and Computation Redistribution for Efficient Face Detection (arXiv:2105.04714)
  - https://arxiv.org/abs/2105.04714
- OpenCV Zoo YuNet model page
  - https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- InsightFace SCRFD repository page
  - https://github.com/deepinsight/insightface/tree/master/detection/scrfd
