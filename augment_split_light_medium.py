import argparse
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create light/medium augmented datasets from data/split/{train,test} with "
            "folder layout compatible with data/split_augmented41mods."
        )
    )
    parser.add_argument("--split-root", default="data/split", help="Input root with train/ and test/ folders.")
    parser.add_argument(
        "--output-root",
        default="data/split_augmented41mods",
        help="Output root: <output>/<level>/<split>/<identity>/<image>__<level>_<idx>.jpg",
    )
    parser.add_argument("--light-per-image", type=int, default=2, help="Light variants per source image.")
    parser.add_argument("--medium-per-image", type=int, default=2, help="Medium variants per source image.")
    parser.add_argument(
        "--include-flip",
        action="store_true",
        help="Allow horizontal flip in augmentations (enable only if mirrored faces are valid).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output root before generation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def clamp_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def apply_brightness_contrast(image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    out = image.astype(np.float32) * contrast
    out += (brightness - 1.0) * 128.0
    return clamp_u8(out)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.arange(256, dtype=np.float32) / 255.0) ** inv
    table = np.clip(table * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def warp_rotate_translate_zoom(
    image: np.ndarray,
    angle_deg: float,
    tx_px: float,
    ty_px: float,
    zoom: float,
) -> np.ndarray:
    h, w = image.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, zoom)
    matrix[0, 2] += tx_px
    matrix[1, 2] += ty_px
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return clamp_u8(image.astype(np.float32) + noise)


def apply_motion_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / float(k)
    return cv2.filter2D(image, -1, kernel)


def crop_and_resize(image: np.ndarray, crop_ratio: float, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    ch = max(8, int(round(h * crop_ratio)))
    cw = max(8, int(round(w * crop_ratio)))
    max_y = max(0, h - ch)
    max_x = max(0, w - cw)
    y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    cropped = image[y0 : y0 + ch, x0 : x0 + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_light_occlusion(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    area_ratio = float(rng.uniform(0.05, 0.10))
    target_area = max(1.0, area_ratio * h * w)
    rect_h = int(max(4, round(math.sqrt(target_area * (h / max(w, 1))))))
    rect_w = int(max(4, round(target_area / max(rect_h, 1))))
    rect_h = min(rect_h, h)
    rect_w = min(rect_w, w)

    forehead_y_max = max(0, int(h * 0.30) - rect_h)
    cheek_y_min = int(h * 0.58)
    cheek_y_max = max(cheek_y_min, min(h - rect_h, int(h * 0.82)))
    use_forehead = bool(rng.integers(0, 2))
    if use_forehead:
        y0 = int(rng.integers(0, forehead_y_max + 1)) if forehead_y_max > 0 else 0
    else:
        y0 = int(rng.integers(cheek_y_min, cheek_y_max + 1)) if cheek_y_max > cheek_y_min else cheek_y_min
        y0 = min(max(0, y0), max(0, h - rect_h))

    x_margin = max(0, int(w * 0.10))
    x_min = x_margin
    x_max = max(x_min, w - rect_w - x_margin)
    x0 = int(rng.integers(x_min, x_max + 1)) if x_max > x_min else x_min
    x0 = min(max(0, x0), max(0, w - rect_w))

    out = image.copy()
    patch = out[y0 : y0 + rect_h, x0 : x0 + rect_w]
    if patch.size == 0:
        return out
    fill = int(np.median(patch))
    alpha = float(rng.uniform(0.45, 0.70))
    blended = (alpha * patch.astype(np.float32)) + ((1.0 - alpha) * fill)
    out[y0 : y0 + rect_h, x0 : x0 + rect_w] = clamp_u8(blended)
    return out


def maybe_flip_horizontal(image: np.ndarray, allow_flip: bool, rng: np.random.Generator) -> np.ndarray:
    if allow_flip and bool(rng.integers(0, 2)):
        return cv2.flip(image, 1)
    return image


def augment_light(image: np.ndarray, include_flip: bool, rng: np.random.Generator) -> np.ndarray:
    out = image.copy()
    out = apply_brightness_contrast(
        out,
        brightness=float(rng.uniform(0.8, 1.2)),
        contrast=float(rng.uniform(0.9, 1.1)),
    )
    h, w = out.shape[:2]
    out = warp_rotate_translate_zoom(
        out,
        angle_deg=float(rng.uniform(-5.0, 5.0)),
        tx_px=float(rng.uniform(-5.0, 5.0)),
        ty_px=float(rng.uniform(-5.0, 5.0)),
        zoom=float(rng.uniform(0.95, 1.05)),
    )
    out = cv2.GaussianBlur(out, (3, 3), 0)
    sigma = float(rng.uniform(5.0, 10.0))
    out = add_gaussian_noise(out, sigma=sigma, rng=rng)
    out = maybe_flip_horizontal(out, allow_flip=include_flip, rng=rng)
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def augment_medium(image: np.ndarray, include_flip: bool, rng: np.random.Generator) -> np.ndarray:
    out = image.copy()
    out = apply_brightness_contrast(
        out,
        brightness=float(rng.uniform(0.6, 1.4)),
        contrast=float(rng.uniform(0.7, 1.3)),
    )
    out = apply_gamma(out, gamma=float(rng.uniform(0.7, 1.5)))
    out = warp_rotate_translate_zoom(
        out,
        angle_deg=float(rng.uniform(-10.0, 10.0)),
        tx_px=float(rng.uniform(-10.0, 10.0)),
        ty_px=float(rng.uniform(-10.0, 10.0)),
        zoom=float(rng.uniform(0.9, 1.1)),
    )
    out = crop_and_resize(out, crop_ratio=float(rng.uniform(0.90, 0.97)), rng=rng)
    out = cv2.GaussianBlur(out, (5, 5), 0)
    if bool(rng.integers(0, 2)):
        out = apply_motion_blur(out, kernel_size=int(rng.integers(3, 6)))
    out = add_gaussian_noise(out, sigma=float(rng.uniform(10.0, 20.0)), rng=rng)
    out = apply_light_occlusion(out, rng=rng)
    out = maybe_flip_horizontal(out, allow_flip=include_flip, rng=rng)
    return out


def collect_split_images(split_dir: Path) -> List[Tuple[str, Path]]:
    if not split_dir.is_dir():
        return []
    items: List[Tuple[str, Path]] = []
    for identity_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for img_path in sorted(identity_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                items.append((identity_dir.name, img_path))
    return items


def ensure_structure(output_root: Path) -> None:
    for level in ("light", "medium"):
        for split_name in ("train", "test"):
            (output_root / level / split_name).mkdir(parents=True, exist_ok=True)


def build_jobs(light_count: int, medium_count: int) -> List[Tuple[str, int]]:
    jobs: List[Tuple[str, int]] = []
    if light_count > 0:
        jobs.append(("light", light_count))
    if medium_count > 0:
        jobs.append(("medium", medium_count))
    return jobs


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    output_root = Path(args.output_root)
    rng = np.random.default_rng(args.seed)

    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing split root: {split_root}")

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    ensure_structure(output_root)

    jobs = build_jobs(args.light_per_image, args.medium_per_image)
    if not jobs:
        raise ValueError("Nothing to generate. Set --light-per-image or --medium-per-image above 0.")

    written: Dict[str, int] = {"light": 0, "medium": 0}
    skipped = 0
    source_images = 0

    for split_name in ("train", "test"):
        split_images = collect_split_images(split_root / split_name)
        print(f"[INFO] {split_name}: found {len(split_images)} source images")
        source_images += len(split_images)
        for identity, src_path in split_images:
            image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[WARN] Could not read {src_path}")
                continue

            stem = src_path.stem
            for level, per_image in jobs:
                out_dir = output_root / level / split_name / identity
                out_dir.mkdir(parents=True, exist_ok=True)
                for idx in range(1, per_image + 1):
                    out_path = out_dir / f"{stem}__{level}_{idx:02d}.jpg"
                    if out_path.exists() and not args.overwrite:
                        skipped += 1
                        continue
                    if level == "light":
                        aug = augment_light(image, include_flip=args.include_flip, rng=rng)
                    else:
                        aug = augment_medium(image, include_flip=args.include_flip, rng=rng)
                    ok = cv2.imwrite(str(out_path), aug)
                    if ok:
                        written[level] += 1
                    else:
                        print(f"[WARN] Failed writing {out_path}")

    print(f"[DONE] Source images: {source_images}")
    print(f"[DONE] Light written: {written['light']}")
    print(f"[DONE] Medium written: {written['medium']}")
    print(f"[DONE] Skipped existing: {skipped}")
    print(f"[DONE] Output root: {output_root.resolve()}")


if __name__ == "__main__":
    main()
