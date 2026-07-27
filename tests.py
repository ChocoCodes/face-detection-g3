import os
import json
import argparse
import numpy as np
from feature_extract import load_database, MODELS
from loader import plot 

# All similarity() methods return a distance -- lower = more similar
MODEL_THRESHOLDS = {
    "mobilenet": 0.70,   # Placeholder -- calibrate manually
    "facenet": 0.80,     # DeepFace official (try 0.80 to 1.10)
    "arcface": 0.65,    # Angular cosine distance (1 - cosine_sim)
    "sface": 1.018,      # 1.128 Validated NormL2 threshold metric, 1.018 = lfw threshold at 165th error pair
}

BATCH_SIZE = 256

def to_features_list(feature_db):
    """
    Flattens the {identity: {"embeddings": [...], "filenames": [...]}} dict
    saved by build_face_db.py into the list-of-records shape the sweep
    below expects: one record per image, not per identity.
    """
    features = []
    for identity, entry in feature_db.items():
        embeddings = entry["embeddings"]
        filenames = entry.get("filenames", [f"{identity}_{i}" for i in range(len(embeddings))])
        for vec, fname in zip(embeddings, filenames):
            features.append({
                "identity": identity,
                "filename": fname,
                "embedding": np.asarray(vec, dtype=np.float32),
            })
            
    return features

def get_feature_metadata(embeddings):
    """
    Programmatically calculates feature dimensions, size in bytes, 
    and checks if it meets the <1 KB target metric constraint.
    """
    if len(embeddings) == 0:
        return 0, 0, "No"
    
    sample_vector = embeddings[0]
    dimension = int(sample_vector.shape[0])
    
    # .nbytes returns the total bytes consumed by the elements of the array
    byte_size = int(sample_vector.nbytes) 
    under_1kb = "Yes" if byte_size < 1024 else "No"
    
    return dimension, byte_size, under_1kb

# ==========================================
# 2. INDEPENDENCE TEST (FP)
# ==========================================
def run_independence(embeddings, identities, model, threshold):
    print("\nInitiating N x (N-1) Independence Test (FP)...")
    n = len(embeddings)
    fp_pairs = []
    total_cross_pairs = 0
    all_pair_distances = []

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = embeddings[start:end]
        batch_identities = identities[start:end]

        distances = compute_distances(embeddings, batch, model)  # (batch, n)

        diff_identity = batch_identities[:, None] != identities[None, :]

        # compute global indexes i and j and extract the upper triangle of the matrix: A->B pair checking only  
        global_i = np.arange(start, end)[:, None]
        global_j = np.arange(n)[None, :]
        upper_tri = global_i < global_j

        mask = diff_identity & upper_tri
        total_cross_pairs += int(mask.sum())

        # Save distances per batch for distribution plot
        all_pair_distances.append(distances[mask].astype(np.float32))

        matched = mask & (distances <= threshold)
        rows, cols = np.where(matched)
        for r, c in zip(rows, cols):
            i = start + r
            fp_pairs.append({
                "name_a": identities[i],
                "name_b": identities[c],
                "distance": float(distances[r, c]),
            })

    all_pair_distances = np.concatenate(all_pair_distances) if all_pair_distances else np.array([], dtype=np.float32)
    return fp_pairs, total_cross_pairs, all_pair_distances


# ==========================================
# 3. (1:1 TP SPACE)
# ==========================================
def run_verification(embeddings, identities, filenames, model):
    """
    Performs 1:1 verification.

    Self-matching is allowed, so the closest match should always be
    the query image itself (distance = 0).
    """
    records = []
    n = len(embeddings)

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = embeddings[start : end]
        # Compute distances from the query to every embedding
        distance_matrix = compute_distances(embeddings, batch, model)

        for local_idx, i in enumerate(range(start, end)):
            distances = distance_matrix[local_idx]
            # Only consider the same identity
            mask = identities == identities[i]
            candidate_indices = np.where(mask)[0]
            best_idx = candidate_indices[
                np.argmin(distances[candidate_indices])
            ]
            records.append({
                "identity": identities[i],
                "query": filenames[i],
                "matched": filenames[best_idx],
                "distance": round(float(distances[best_idx]), 6),
                "correct": i == best_idx
            })

    return records

# ==========================================
# 4. SUMMARY
# ==========================================
def summarize(model_label, db_label, fp_pairs, total_cross_pairs, oto, threshold, n_persons, embeddings):
    tp_correct = sum(1 for p in oto if p["correct"])
    fp_percent = round(len(fp_pairs) / total_cross_pairs * 100, 4) if total_cross_pairs else 0.0
    tp_percent = round(tp_correct / len(oto) * 100, 4) if oto else 0.0

    # Programmatically calculate bytes
    dimension, byte_size, under_1kb = get_feature_metadata(embeddings)

    print(f"\n[{model_label.upper()}]")
    print(f"   Cross-identity pairs : {total_cross_pairs}")
    print(f"   False positives      : {len(fp_pairs)} ({fp_percent}%)")
    print(f"   Same-identity pairs  : {len(oto)}")
    print(f"   True positive rate   : {tp_percent}%")
    print(f"   Feature Dimension    : {dimension}")
    print(f"   Feature Size (Bytes) : {byte_size} Bytes")
    print(f"   Under 1 KB Feature?  : {under_1kb}")

    return {
        "DB": db_label,
        "n_persons": n_persons,
        "n_comparisons": total_cross_pairs,
        "n_fp": len(fp_pairs),
        "tp_percent": tp_percent,
        "fp_percent": fp_percent,
        "feature_dimension": dimension,
        "feature_size_bytes": byte_size,
        "under_1kb": under_1kb
    }

def compute_distances(embeddings, batch, model):
    if model == 'arcface':
        return 1.0 - (batch @ embeddings.T)

    batch_sq = np.sum(batch ** 2, axis=1, keepdims=True)   # (batch, 1)
    emb_sq = np.sum(embeddings ** 2, axis=1)[None, :]       # (1, n)
    cross_term = batch @ embeddings.T                        # (batch, n)
    sq_dist = np.maximum(batch_sq + emb_sq - 2 * cross_term, 0)  # clip tiny negatives from float error
    return np.sqrt(sq_dist)

def extract_meta(db_path: str):
    """ Extract database and model information from the db_path provided from the CLI. """
    base = os.path.basename(db_path)
    db_file, _ = os.path.splitext(base)
    db, model = db_file.split('-', 1)

    return db, model

# ==========================================
# 5. ENTRY POINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Run 1:1 and cross-identity independence tests on a pre-built .npy feature database."
    )
    parser.add_argument("--db", required=True, help="Path to the .npy feature database (from db-name.py)")
    parser.add_argument('--generate-plot', action='store_true', help='Generate a frequency vs distance graph from independence test.')

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/images", exist_ok=True)
    os.makedirs("results/features", exist_ok=True)
    
    db, model = extract_meta(args.db)

    output_file = f"{db}-{model}.json"
    output_path = os.path.join("results", output_file)


    print(f"Loading feature database: {args.db}")
    feature_db = load_database(args.db)
    features_list = to_features_list(feature_db)
    print(f"Loaded {len(features_list)} embeddings across {len(feature_db)} identities.")

    embeddings = np.stack([identity['embedding'] for identity in features_list])
    identities = np.array([identity['identity'] for identity in features_list])
    filenames = np.array([item["filename"] for item in features_list])

    if model in ("arcface", "sface"):
        embeddings /= np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )
    # Only instantiate the ONE model needed -- purely for its similarity()
    # method, no re-extraction or image loading happens here.
    threshold = MODEL_THRESHOLDS[model]

    oto = run_verification(
        embeddings,
        identities,
        filenames,
        model
    )

    fp_pairs, total_cross_pairs, all_pair_distances = run_independence(
        embeddings, 
        identities, 
        model, 
        threshold
    )

    summary = summarize(model, db, fp_pairs, total_cross_pairs, oto, threshold, len(feature_db), embeddings)

    print(f"\nWriting structured JSON analysis log to: {output_path}")
    with open(output_path, "w") as f:
        json.dump({model: summary}, f, indent=4)

    # Conditionally generate plot
    if args.generate_plot:
        plot_path = os.path.join('results/images', f"{db}-{model}-distribution.png")
        plot(
            all_pair_distances,
            threshold,
            model,
            output_path=plot_path,
        )

    print("Independence test successfully generated.")


if __name__ == "__main__":
    main()