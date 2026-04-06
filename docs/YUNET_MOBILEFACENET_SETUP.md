# YuNet + MobileFaceNet Pipeline

This document explains how to train, evaluate, and benchmark the new embedding-based pipeline.

## Expected Model Files

Place ONNX files here:

- `models/yunet_mobilefacenet/face_detection_yunet_2023mar.onnx`
- `models/yunet_mobilefacenet/mobilefacenet.onnx`

Notes:
- YuNet ONNX can be obtained from OpenCV Zoo.
- MobileFaceNet ONNX should output a face embedding vector (commonly 128 or 512 dim).

## 1) Train Enrollment

This does not train model weights. It builds identity centroids from your dataset.

```powershell
python src/yunet_mobilefacenet/trainer.py --include-raw --assume-processed-are-cropped
```

Output:

- `models/yunet_mobilefacenet/enrollment.json`

## 2) Evaluate

Quick evaluation:

```powershell
python src/yunet_mobilefacenet/evaluate.py --max-images-per-person 5 --aug-splits original,light --assume-processed-are-cropped --report-json reports/evaluation/yunet_mobilefacenet_quick.json
```

Full evaluation:

```powershell
python src/yunet_mobilefacenet/evaluate.py --max-images-per-person 0 --aug-splits original,light,medium,heavy --assume-processed-are-cropped --report-json reports/evaluation/yunet_mobilefacenet_full.json
```

## 3) Benchmark Against LBPH

Quick comparison report:

```powershell
python src/benchmark/compare_models.py --mode quick --assume-processed-are-cropped
```

Full comparison report:

```powershell
python src/benchmark/compare_models.py --mode full --assume-processed-are-cropped
```

Generated outputs:

- `reports/benchmark/comparison_summary.json`
- `reports/benchmark/comparison_summary.md`

## Reporting Strategy

Compare these metrics first:

- overall hit rate
- per-bucket hit rate (raw, processed, augmented/*)
- elapsed time

If raw hit rate is much lower than processed/augmented, tune:

1. enrollment coverage (`--include-raw`)
2. threshold (`--threshold` in evaluator)
3. detector confidence (`--det-score-threshold`)

## 4) Live Webcam Inference

```powershell
python src/yunet_mobilefacenet/face_detect.py --largest-face-only
```

Notes:
- Press `d` to exit.
- By default, the script uses `recommended_threshold` from `enrollment.json`.
