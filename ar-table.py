import os
import json
import argparse
import numpy as np

# Existing thresholds from your codebase
MODEL_THRESHOLDS = {
    "mobilenet": 0.70,   
    "facenet": 0.80,     
    "arcface": 0.65,    
    "sface": 1.128,      
}

# Fully validated mapping of your 41 modifications (Light, Medium, Heavy)
MODIFICATION_MAPS = {
    "brightness_p25": ("Brightness change", "heavy"),
    "brightness_p20": ("Brightness change", "medium"),
    "brightness_p10": ("Brightness change", "light"),
    "monochrome": ("Colour to monochrome", "light"),
    "jpeg_q30": ("JPEG compression", "heavy"),
    "jpeg_q60": ("JPEG compression", "medium"),
    "jpeg_q80": ("JPEG compression", "light"),
    "color_8bit": ("Colour reduction", "medium"),
    "rgb565": ("Colour reduction", "light"),
    "gaussnoise_12": ("Gaussian noise", "heavy"),
    "gaussnoise_8": ("Gaussian noise", "medium"),
    "gaussnoise_4": ("Gaussian noise", "light"),
    "histeq": ("Histogram equalization", "light"),
    "autolevel": ("Auto-levels", "light"),
    "blur_7": ("Blur", "heavy"),
    "blur_5": ("Blur", "medium"),
    "blur_3": ("Blur", "light"),
    "rot_270": ("Rotation", "heavy"),
    "rot_180": ("Rotation", "medium"),
    "rot_90": ("Rotation", "light"),
    "rot_45": ("Simple rotation", "heavy"),
    "rot_25": ("Simple rotation", "medium"),
    "rot_10": ("Simple rotation", "light"),
    "scale_50": ("Scaling", "heavy"),
    "scale_70": ("Scaling", "medium"),
    "scale_90": ("Scaling", "light"),
    "trans_30": ("Translation", "heavy"),
    "trans_20": ("Translation", "medium"),
    "trans_10": ("Translation", "light"),
    "flip_lr": ("Flip", "light"),
    "aspect_6_3": ("Aspect ratio change", "medium"),
    "aspect_16_9": ("Aspect ratio change", "light"),
    "crop_60": ("Crop", "heavy"),
    "crop_70": ("Crop", "medium"),
    "crop_90": ("Crop", "light"),
    "skew_10": ("Skew", "heavy"),
    "skew_6": ("Skew", "medium"),
    "skew_4": ("Skew", "light"),
    "persp_10": ("Perspective", "heavy"),
    "persp_6": ("Perspective", "medium"),
    "persp_4": ("Perspective", "light"),
}

def load_database(path):
    if path is None:
        raise ValueError("Path is not specified.")
    return np.load(path, allow_pickle=True).item()

def parse_filename(filename):
    """
    Splits 'dark_right__rot_90.jpg' -> ('dark_right', 'rot_90')
    and normalizes 'dark_up_orig.jpg' or 'dark_up__orig.jpg' -> ('dark_up', 'orig')
    """
    # Strip directory path if present, leaving only the filename
    filename = os.path.basename(filename)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Standardize baseline/original files ending with _orig or __orig
    if name_without_ext.endswith("_orig"):
        if name_without_ext.endswith("__orig"):
            base = name_without_ext[:-6]  # strip '__orig'
        else:
            base = name_without_ext[:-5]  # strip '_orig'
        return base, "orig"
        
    if "__" in name_without_ext:
        base, suffix = name_without_ext.split("__", 1)
        return base, suffix
        
    return name_without_ext, None

def compute_distances(embeddings, batch, model):
    if model == 'arcface':
        return 1.0 - (batch @ embeddings.T)
    batch_sq = np.sum(batch ** 2, axis=1, keepdims=True)
    emb_sq = np.sum(embeddings ** 2, axis=1)[None, :]
    cross_term = batch @ embeddings.T
    sq_dist = np.maximum(batch_sq + emb_sq - 2 * cross_term, 0)
    return np.sqrt(sq_dist)

def run_modifications_test(feature_db, model, threshold, ref_db=None):
    """Runs evaluation across all 41 modifications, mapping each modified 
    image back to its respective pose-specific reference.
    """
    print("\nEvaluating Accuracy Ratio (AR) across modifications...")
    
    unique_categories = sorted(list(set(val[0] for val in MODIFICATION_MAPS.values())))
    results = {
        cat: {"heavy": {"K": 0, "M": 0}, "medium": {"K": 0, "M": 0}, "light": {"K": 0, "M": 0}} 
        for cat in unique_categories
    }

    # 1. Build Reference Database
    global_reference_embeddings = {}
    
    # If a separate reference DB was loaded (e.g., LaSalleDB1), catalog those first
    if ref_db:
        print("Cataloging original references from LaSalleDB1...")
        for identity, entry in ref_db.items():
            embeddings = entry["embeddings"]
            filenames = entry["filenames"]
            identity_refs = {}
            for idx, fname in enumerate(filenames):
                base, suffix = parse_filename(fname)
                if suffix == "orig":
                    emb = np.asarray(embeddings[idx], dtype=np.float32)
                    if model in ("arcface", "sface"):
                        emb /= np.linalg.norm(emb)
                    identity_refs[base] = emb
            global_reference_embeddings[identity] = identity_refs

    # 2. Process DB2 (Modifications)
    total_mod_images_processed = 0
    total_references_matched = 0

    for identity, entry in feature_db.items():
        embeddings = entry["embeddings"]
        filenames = entry["filenames"]
        
        # Pull references for this identity
        reference_embeddings = global_reference_embeddings.get(identity, {})
        modified_embeddings = []

        # Parse local filenames
        for idx, fname in enumerate(filenames):
            base, suffix = parse_filename(fname)
            emb = np.asarray(embeddings[idx], dtype=np.float32)
            
            if model in ("arcface", "sface"):
                emb /= np.linalg.norm(emb)

            if suffix == "orig":
                reference_embeddings[base] = emb
            else:
                modified_embeddings.append((base, suffix, emb))

        total_mod_images_processed += len(modified_embeddings)

        # Match each modified image with its specific pose reference
        for base, suffix, mod_emb in modified_embeddings:
            if suffix not in MODIFICATION_MAPS:
                continue
            
            category, level = MODIFICATION_MAPS[suffix]
            
            if base in reference_embeddings:
                total_references_matched += 1
                ref_emb = reference_embeddings[base]
                dist = compute_distances(mod_emb[None, :], ref_emb[None, :], model)[0, 0]
                
                is_match = dist <= threshold
                
                results[category][level]["M"] += 1
                if is_match:
                    results[category][level]["K"] += 1

    print(f"Total modified images processed: {total_mod_images_processed}")
    print(f"Total successful reference matches: {total_references_matched}")

    if total_references_matched == 0:
        print("\n[ERROR] No matching original references were found for your modified images!")
        print("Please ensure your original reference images are either in this file or that 'lasalledb1-sface.npy' is in the same directory.")
        return

    # RESULTS
    print("\n" + "="*110)
    print(f" ACCURACY RATIO (AR) REPORT FOR LaSalleDB2 ({model.upper()})")
    print(f" Decision Threshold: {threshold}")
    print("="*110)
    print(f"{'Modified Items':<30} | {'Heavy % (K/M)':<20} | {'Medium % (K/M)':<20} | {'Light % (K/M)':<20} | {'Mean %':<10}")
    print("-"*110)

    all_means = []

    for cat in unique_categories:
        row_rates = []
        row_strs = {}
        
        for level in ["heavy", "medium", "light"]:
            k = results[cat][level]["K"]
            m = results[cat][level]["M"]
            if m > 0:
                rate = (k / m) * 100
                row_rates.append(rate)
                row_strs[level] = f"{rate:>6.2f}% ({k}/{m})"
            else:
                row_strs[level] = "   N/A"

        mean_rate = np.mean(row_rates) if row_rates else 0.0
        if row_rates:
            all_means.append(mean_rate)
            
        mean_str = f"{mean_rate:>6.2f}%" if row_rates else "   N/A"
        print(f"{cat:<30} | {row_strs['heavy']:<20} | {row_strs['medium']:<20} | {row_strs['light']:<20} | {mean_str:<10}")

    overall_average = np.mean(all_means) if all_means else 0.0
    print("-"*110)
    print(f"{'OVERALL AVERAGE ACCURACY':<30} | {' ':<20} | {' ':<20} | {' ':<20} | {overall_average:>6.2f}%")
    print("="*110)

def main():
    parser = argparse.ArgumentParser(description="Run accuracy tests on an existing .npy feature database.")
    parser.add_argument("--db", required=True, help="Path to your .npy feature database.")
    args = parser.parse_args()

    # Automatically parse model and database names from filename path
    base = os.path.basename(args.db)
    db_file, _ = os.path.splitext(base)
    db_name, model_name = db_file.split('-', 1)

    print(f"Loading database: {args.db}")
    feature_db = load_database(args.db)
    threshold = MODEL_THRESHOLDS[model_name]

    # --- 1. RUN DIAGNOSTICS ---
    print("\n--- DATABASE DIAGNOSTICS ---")
    first_identity = list(feature_db.keys())[0]
    print(f"Total identities found: {len(feature_db)}")
    print(f"Sample identity: '{first_identity}'")
    print(f"Sample filenames in database (first 5):")
    sample_filenames = feature_db[first_identity]["filenames"][:5]
    for fname in sample_filenames:
        base_parsed, suffix_parsed = parse_filename(fname)
        print(f"  - Original File: '{fname}' -> Parsed as: (Base: '{base_parsed}', Suffix: '{suffix_parsed}')")
    
    # Check if there are any original files locally inside this database
    has_orig = any(parse_filename(f)[1] == "orig" for f in feature_db[first_identity]["filenames"])
    print(f"Local reference images (_orig) present inside loaded file? {has_orig}")
    print("----------------------------\n")

    # --- 2. DYNAMIC REFERENCE LOADING ---
    ref_db = None
    if not has_orig and "lasalledb2" in db_name.lower():
        # Look for DB1 companion file
        dir_name = os.path.dirname(args.db)
        db1_filename = f"lasalledb1-{model_name}.npy"
        db1_path = os.path.join(dir_name, db1_filename) if dir_name else db1_filename
        
        if os.path.exists(db1_path):
            print(f"Detected LaSalleDB2. Automatically loading companion baseline file: {db1_path}")
            ref_db = load_database(db1_path)
        else:
            print(f"[WARNING] No reference images found inside your file, and companion database '{db1_filename}' was not found in the same folder.")

    # Run the AR table evaluation
    run_modifications_test(feature_db, model_name, threshold, ref_db=ref_db)

if __name__ == "__main__":
    main()