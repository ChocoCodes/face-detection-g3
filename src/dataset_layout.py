import os
from typing import List, Tuple


KNOWN_SPLIT_NAMES = {"train", "test"}


def infer_target_split_name(raw_dir: str, processed_dir: str) -> str | None:
    for candidate in (processed_dir, raw_dir):
        base = os.path.basename(os.path.normpath(candidate)).lower()
        if base in KNOWN_SPLIT_NAMES:
            return base
    return None


def gather_augmented_person_dirs(
    augmented_root: str,
    aug_splits: set[str],
    target_split: str | None,
) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    if not os.path.isdir(augmented_root):
        return out

    for split_name in sorted(os.listdir(augmented_root)):
        split_lower = split_name.lower()
        if aug_splits and split_lower not in aug_splits:
            continue

        split_path = os.path.join(augmented_root, split_name)
        if not os.path.isdir(split_path):
            continue

        nested_split_names: List[str] = []
        if target_split:
            candidate = os.path.join(split_path, target_split)
            if os.path.isdir(candidate):
                nested_split_names = [target_split]
        else:
            for name in sorted(os.listdir(split_path)):
                name_lower = name.lower()
                nested_path = os.path.join(split_path, name)
                if name_lower in KNOWN_SPLIT_NAMES and os.path.isdir(nested_path):
                    nested_split_names.append(name_lower)

        if nested_split_names:
            for nested_name in nested_split_names:
                nested_path = os.path.join(split_path, nested_name)
                bucket = f"augmented/{split_name}" if target_split else f"augmented/{split_name}/{nested_name}"
                for person in sorted(os.listdir(nested_path)):
                    person_path = os.path.join(nested_path, person)
                    if os.path.isdir(person_path):
                        out.append((bucket, person, person_path))
            continue

        bucket = f"augmented/{split_name}"
        for person in sorted(os.listdir(split_path)):
            person_path = os.path.join(split_path, person)
            if os.path.isdir(person_path):
                out.append((bucket, person, person_path))

    return out
