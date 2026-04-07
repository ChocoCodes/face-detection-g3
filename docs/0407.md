# Changelog - 2026-04-07

## Summary
Today we standardized dataset defaults across training/evaluation scripts and added live FPS logging + aggregation for webcam detection pipelines.

## 1) Dataset Default Behavior Cleanup
Goal: trainers use augmented data by default, evaluators use raw data by default.

### Training scripts
- Updated trainer defaults so processed/raw are opt-in where applicable.
- Added opt-in processed flags and kept raw opt-in behavior where already present.

Files changed:
- src/arcface/trainer.py
- src/mobilefacenet/trainer.py
- src/edgeface/trainer.py
- src/lbph/trainer.py
- src/edgeface/common.py

### Evaluation scripts
- Updated evaluator defaults so they evaluate raw dataset by default.
- Added opt-in flags for processed and augmented datasets.

Files changed:
- src/arcface/evaluate.py
- src/mobilefacenet/evaluate.py
- src/edgeface/evaluate.py
- src/lbph/evaluate.py

### Benchmark compatibility update
- Updated benchmark runner to explicitly include processed + augmented datasets so comparisons remain broad after evaluator defaults changed.

File changed:
- src/benchmark/compare_models.py

## 2) Live Webcam FPS Logging + Aggregation
Goal: every live detector writes FPS logs and a per-run summary, then aggregate average FPS per algorithm.

### Added FPS logging options to live detectors
Each script now supports:
- --fps-log-path
- --fps-log-interval
- --fps-summary-dir
- --disable-fps-log

Each run now writes:
- periodic JSONL FPS samples
- per-run summary JSON with average_fps, frames, duration_seconds

Files changed:
- src/arcface/detect.py
- src/mobilefacenet/detect.py
- src/lbph/detect.py
- src/edgeface/face_detect.py

### Added FPS aggregation utility
- Added script to aggregate average FPS by algorithm across run summary files.

File added:
- src/benchmark/aggregate_live_fps.py

Default output:
- reports/benchmark/live_fps/aggregate_summary.json

## 3) Notes
- All touched Python files were checked and reported no diagnostics errors after changes.
- Existing user-created docs/GOAL.md was left unchanged.
