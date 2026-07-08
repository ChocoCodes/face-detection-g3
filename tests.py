import os
import json
import argparse
import numpy as np
from feature_extract import load_database, MODELS

# All similarity() methods return a distance -- lower = more similar
MODEL_THRESHOLDS = {
    "mobilenet": 0.70,   # Placeholder -- calibrate manually
    "facenet": 0.80,     # DeepFace official (try 0.80 to 1.10)
    "arcface": 0.65,    # Angular cosine distance (1 - cosine_sim)
    "sface": 1.128,      # Validated NormL2 threshold metric
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

# ==========================================
# 2. INDEPENDENCE TEST (FP SPACE)
# ==========================================
def run_cross_identity(embeddings, identities, model, threshold):
    print("\nInitiating N x (N-1) Independence Test (FP)...")
    n = len(embeddings)
    fp_pairs = []
    total_cross_pairs = 0

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = embeddings[start:end]
        batch_identities = identities[start:end]

        distances = compute_distances(embeddings, batch, model)  # (batch, n)

        diff_identity = batch_identities[:, None] != identities[None, :]
        total_cross_pairs += int(diff_identity.sum())

        matched = diff_identity & (distances <= threshold)
        rows, cols = np.where(matched)
        for r, c in zip(rows, cols):
            i = start + r
            fp_pairs.append({
                "name_a": identities[i],
                "name_b": identities[c],
                "distance": float(distances[r, c]),
            })

    return fp_pairs, total_cross_pairs


# ==========================================
# 3. (1:1 TP SPACE)
# ==========================================
def run_one_to_one(embeddings, identities, filenames, model):
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
def summarize(model_label, db_label, fp_pairs, total_cross_pairs, oto, threshold, n_persons):
    tp_correct = sum(1 for p in oto if p["correct"])
    fp_percent = round(len(fp_pairs) / total_cross_pairs * 100, 4) if total_cross_pairs else 0.0
    tp_percent = round(tp_correct / len(oto) * 100, 4) if oto else 0.0

    print(f"\n[{model_label}]")
    print(f"   Cross-identity pairs : {total_cross_pairs}")
    print(f"   False positives      : {len(fp_pairs)} ({fp_percent}%)")
    print(f"   Same-identity pairs  : {len(oto)}")
    print(f"   True positive rate   : {tp_percent}%")

    return {
        "DB": db_label,
        "n_persons": n_persons,
        "n_comparisons": total_cross_pairs,
        "n_fp": len(fp_pairs),
        "tp_percent": tp_percent,
        "fp_percent": fp_percent,
    }


def compute_distances(embeddings, batch, model):
    if model == 'arcface':
        return 1.0 - (batch @ embeddings.T)

    batch_sq = np.sum(batch ** 2, axis=1, keepdims=True)   # (batch, 1)
    emb_sq = np.sum(embeddings ** 2, axis=1)[None, :]       # (1, n)
    cross_term = batch @ embeddings.T                        # (batch, n)
    sq_dist = np.maximum(batch_sq + emb_sq - 2 * cross_term, 0)  # clip tiny negatives from float error
    return np.sqrt(sq_dist)

# ==========================================
# 5. ENTRY POINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Run 1:1 and cross-identity independence tests on a pre-built .npy feature database."
    )
    parser.add_argument("--db", required=True, help="Path to the .npy feature database (from build_face_db.py)")
    parser.add_argument("--model", required=True, choices=list(MODELS),
                         help="Which model this .npy was built with (needed for its similarity() + threshold)")
    parser.add_argument("--output", default="results.json", help="Output JSON filename")
    args = parser.parse_args()

    output_file = args.output if args.output.lower().endswith(".json") else args.output + ".json"
    output_path = os.path.join("results", output_file)
    os.makedirs("results", exist_ok=True)

    print(f"Loading feature database: {args.db}")
    feature_db = load_database(args.db)
    features_list = to_features_list(feature_db)
    print(f"Loaded {len(features_list)} embeddings across {len(feature_db)} identities.")

    embeddings = np.stack([identity['embedding'] for identity in features_list])
    identities = np.array([identity['identity'] for identity in features_list])
    filenames = np.array([item["filename"] for item in features_list])

    if args.model in ("arcface", "sface"):
        embeddings /= np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )
    # Only instantiate the ONE model needed -- purely for its similarity()
    # method, no re-extraction or image loading happens here.
    # model = MODELS[args.model]()
    threshold = MODEL_THRESHOLDS[args.model]

    oto = run_one_to_one(
        embeddings,
        identities,
        filenames,
        args.model
    )

    fp_pairs, total_cross_pairs = run_cross_identity(
        embeddings, 
        identities, 
        args.model, 
        threshold
    )

    summary = summarize(args.model, args.db, fp_pairs, total_cross_pairs, oto, threshold, len(feature_db))

    print(f"\nWriting structured JSON analysis log to: {output_path}")
    with open(output_path, "w") as f:
        json.dump({args.model: summary}, f, indent=4)

    print("Independence test successfully generated.")


if __name__ == "__main__":
    main()