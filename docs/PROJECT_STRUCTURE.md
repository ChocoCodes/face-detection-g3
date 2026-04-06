# Project Structure

This repository is organized by responsibility:

- `src/lbph/`
  - `trainer.py`: trains the LBPH recognizer
  - `face_detect.py`: webcam inference using the trained LBPH model
  - `evaluate.py`: offline dataset evaluation with hit-rate and JSON report output
- `src/yunet_mobilefacenet/`
  - `trainer.py`: builds embedding enrollment (identity centroids)
  - `evaluate.py`: evaluates YuNet + MobileFaceNet hit-rate on dataset buckets
  - `face_detect.py`: webcam inference using YuNet detection + MobileFaceNet matching
- `src/edgeface/`
  - `trainer.py`: builds embedding enrollment using YuNet + EdgeFace
  - `evaluate.py`: evaluates YuNet + EdgeFace hit-rate on dataset buckets
  - `face_detect.py`: webcam inference using YuNet detection + EdgeFace matching
- `src/benchmark/`
  - `compare_models.py`: runs LBPH and embedding evaluators and writes comparison reports
- `src/arcface_mobilenet_int8/`
  - `quantize_model.py`: builds INT8 ArcFace model pack in a separate models folder
  - `face_detect.py`: wrapper for live detection using INT8 model directory defaults
  - `trainer.py`: wrapper for enrollment build using INT8 model directory defaults
  - `evaluate.py`: wrapper for evaluation using INT8 model directory defaults
- `models/lbph/`
  - `trainer_lasalle.yml`: trained LBPH model artifact
  - `labels_lasalle.json`: label map artifact
- `models/yunet_mobilefacenet/`
  - `face_detection_yunet_2023mar.onnx`: YuNet detector model
  - `mobilefacenet.onnx`: embedding model
  - `enrollment.json`: enrollment centroids and metadata
- `models/edgeface/`
  - `face_detection_yunet_2023mar.onnx`: YuNet detector model expected by the EdgeFace pipeline
  - `edgeface_xs.onnx`: EdgeFace recognizer model expected by the pipeline
  - `enrollment.json`: generated enrollment centroids and metadata
- `models/arcface_mobilenet_int8/`
  - quantized ArcFace model pack (generated via `quantize_model.py`)
- `reports/evaluation/`
  - evaluation outputs (`.json`, `.txt`)
- `reports/benchmark/`
  - model-vs-model comparison outputs (`.json`, `.md`)
- `data/`
  - datasets (`lasalle_db1`, `lasalle_db1_processed`, `augmented41mods`, etc.)
- `docs/`
  - documentation and model notes
- `archive/`
  - legacy or previous script versions
- `haar/`
  - Haar cascade XML resources

## Recommended Commands

Train:

```powershell
python src/lbph/trainer.py
```

Evaluate:

```powershell
python src/lbph/evaluate.py --report-json reports/evaluation/lbph_eval.json
```

Evaluate (YuNet + MobileFaceNet):

```powershell
python src/yunet_mobilefacenet/evaluate.py --report-json reports/evaluation/yunet_mobilefacenet_eval.json
```

Compare models:

```powershell
python src/benchmark/compare_models.py --mode quick --assume-processed-are-cropped
```

Live detect:

```powershell
python src/lbph/face_detect.py --largest-face-only
```

Live detect (YuNet + MobileFaceNet):

```powershell
python src/yunet_mobilefacenet/face_detect.py --largest-face-only
```

Live detect (YuNet + EdgeFace):

```powershell
python src/edgeface/face_detect.py --largest-face-only
```

All scripts are project-root aware, so they can be run from any working directory.
