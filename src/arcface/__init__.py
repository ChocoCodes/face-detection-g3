"""ArcFace + robust backbone training and evaluation for diverse real-world conditions.

Supports multiple ArcFace embedding models:
- InsightFace buffalo_s (balanced speed/accuracy)
- InsightFace r50 (maximum accuracy for fixed roster)
- MobileNet v3 + ArcFace (ultra-lightweight)

Designed for robustness across:
- Variable lighting (indoor, outdoor, low-light)
- Distance variations (close-up to far field)
- Weather/environment changes
- Pose and expression variation
"""
