import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _stable_slug(raw: str, prefix: str, length: int = 12) -> str:
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def _clean_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "default"


def relativize_path(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def build_dataset_profile(
    *,
    base_data_dir: str,
    raw_dir_name: str,
    include_raw: bool,
    processed_dir_name: str,
    include_processed: bool,
    augmented_dir_name: str,
    include_augmented: bool,
    aug_splits: Iterable[str],
    target_split: str | None,
) -> dict:
    aug_list = sorted({str(s).strip().lower() for s in aug_splits if str(s).strip()})
    profile = {
        "base_data_dir": relativize_path(base_data_dir),
        "raw_dir_name": raw_dir_name,
        "include_raw": bool(include_raw),
        "processed_dir_name": processed_dir_name,
        "include_processed": bool(include_processed),
        "augmented_dir_name": augmented_dir_name,
        "include_augmented": bool(include_augmented),
        "aug_splits": aug_list,
        "target_split": target_split or "",
    }
    key_source = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    dataset_key = _stable_slug(key_source, prefix="ds")

    tokens: list[str] = []
    if include_raw:
        tokens.append(f"raw={raw_dir_name}")
    if include_processed:
        tokens.append(f"processed={processed_dir_name}")
    if include_augmented:
        joined = ",".join(aug_list) if aug_list else "all"
        tokens.append(f"aug={augmented_dir_name}[{joined}]")
    if target_split:
        tokens.append(f"split={target_split}")
    label = " | ".join(tokens) if tokens else "no_dataset_selected"

    profile["dataset_key"] = dataset_key
    profile["label"] = label
    return profile


def derive_model_variant(*artifact_paths: str, fallback: str = "default") -> str:
    tokens: list[str] = []
    for value in artifact_paths:
        if not value:
            continue
        stem = Path(value).stem
        if stem:
            tokens.append(_clean_token(stem))
    if not tokens:
        return _clean_token(fallback)
    return "__".join(tokens)


def attach_entity_identity(
    *,
    report: dict,
    model_family: str,
    dataset_profile: dict,
    model_variant: str,
    run_tag: str = "",
) -> dict:
    clean_model = _clean_token(model_family.lower())
    clean_variant = _clean_token(model_variant)
    clean_tag = _clean_token(run_tag) if run_tag else ""
    dataset_key = str(dataset_profile.get("dataset_key", ""))
    if not dataset_key:
        dataset_key = _stable_slug(json.dumps(dataset_profile, sort_keys=True), prefix="ds")
    entity_key = f"{clean_model}::{clean_variant}::{dataset_key}"
    if clean_tag:
        entity_key = f"{entity_key}::{clean_tag}"

    report["model_family"] = model_family
    report["model_variant"] = clean_variant
    report["dataset_profile"] = dataset_profile
    report["dataset_key"] = dataset_key
    report["run_tag"] = clean_tag
    report["entity_key"] = entity_key
    return report
