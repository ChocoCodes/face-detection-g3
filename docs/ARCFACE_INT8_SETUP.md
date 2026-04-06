# ArcFace INT8 Setup

This is a parallel pipeline that keeps your original ArcFace files untouched.

## 1) Build INT8 model pack

```powershell
python src/arcface_mobilenet_int8/quantize_model.py
```

Default output:

- `models/arcface_mobilenet_int8/`

## 2) Run live webcam (INT8 model pack)

```powershell
python src/arcface_mobilenet_int8/face_detect.py
```

## 3) Rebuild enrollment (optional, recommended)

```powershell
python src/arcface_mobilenet_int8/trainer.py --include-raw
```

## 4) Evaluate

```powershell
python src/arcface_mobilenet_int8/evaluate.py --report-json reports/evaluation/arcface_int8_eval.json
```

