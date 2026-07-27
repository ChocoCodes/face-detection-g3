import os
import numpy as np
import argparse
from feature_extract import load_database
from tests import extract_meta, compute_distances

BATCH_SIZE = 256


def filter_to_first_photo(feature_db, db):
    """
    Filters the feature database down to one photo per identity.
    LFW Database = <identity>_0001.jpg
    LaSalle Database 1 & 2 = light_front__orig.jpg
    """
    filtered_list = []
    for identity, entry in feature_db.items():
        embeddings = entry["embeddings"]
        filenames = entry["filenames"]
        target_filename = f"{identity}_0001.jpg" if db == "lfw" else "light_front__orig.jpg"

        if target_filename in filenames:
            idx = filenames.index(target_filename)
            filtered_list.append({
                "identity": identity,
                "filename": filenames[idx],
                "embedding": np.asarray(embeddings[idx], dtype=np.float32),
            })
        else:
            print(f"Warning: '{target_filename}' not found for identity '{identity}'")

    return filtered_list


def find_top_k_impostor_pairs(embeddings, identities, filenames, model, k):
    """
    Single vectorized pass (same batching pattern as run_independence) that
    tracks BOTH the full impostor distance array AND which (i, j) pair each
    distance belongs to -- so the top-k closest impostor pairs can be
    recovered directly afterward with a simple argpartition, with no need
    to re-loop over every pair a second time in pure Python.
    """
    n = len(embeddings)
    all_distances = []
    all_i = []
    all_j = []

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = embeddings[start:end]
        batch_identities = identities[start:end]

        distances = compute_distances(embeddings, batch, model)  # (batch, n)

        diff_identity = batch_identities[:, None] != identities[None, :]
        global_i = np.arange(start, end)[:, None]
        global_j = np.arange(n)[None, :]
        upper_tri = global_i < global_j

        mask = diff_identity & upper_tri
        rows, cols = np.where(mask)

        all_distances.append(distances[rows, cols].astype(np.float32))
        all_i.append((start + rows).astype(np.int32))
        all_j.append(cols.astype(np.int32))

    all_distances = np.concatenate(all_distances)
    all_i = np.concatenate(all_i)
    all_j = np.concatenate(all_j)
    total_pairs = len(all_distances)

    k = min(k, total_pairs)
    # argpartition is O(n) -- avoids a full sort just to get the smallest k
    top_k_idx = np.argpartition(all_distances, k - 1)[:k]
    top_k_idx = top_k_idx[np.argsort(all_distances[top_k_idx])]  # sort just this small subset

    top_k_pairs = [
        {
            "name_a": identities[all_i[idx]],
            "name_b": identities[all_j[idx]],
            "filename_a": filenames[all_i[idx]],
            "filename_b": filenames[all_j[idx]],
            "distance": float(all_distances[idx]),
        }
        for idx in top_k_idx
    ]

    return top_k_pairs, total_pairs, all_distances


def main():
    parser = argparse.ArgumentParser(
        description="Determine a ppm-based threshold from an independence test on a 1-photo-per-identity subset."
    )
    parser.add_argument("--db", required=True, help="Path to your pre-built .npy feature database.")
    parser.add_argument("--ppm", type=float, default=1.0,
                         help="Target False Alarm Rate in parts-per-million (e.g. 1 for LFW1's 1ppm target, "
                              "100 for LaSalleDB1's 100ppm target). Default: 1.")
    args = parser.parse_args()

    print(f"Loading database from: {args.db}")
    # Load and extract the feature db and model name from "features/db-model.npy" file format
    feature_db = load_database(args.db)
    db, model = extract_meta(args.db)

    filtered_list = filter_to_first_photo(feature_db, db)
    print(f"Filtered database down to {len(filtered_list)} identities (1 photo each).")

    if len(filtered_list) < 2:
        print("Error: Not enough identities found with a '_0001.jpg' photo.")
        return

    embeddings = np.stack([item['embedding'] for item in filtered_list])
    identities = np.array([item['identity'] for item in filtered_list])
    filenames = np.array([item['filename'] for item in filtered_list])

    if model in ("arcface", "sface"):
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    print("Calculating distances for all cross-identity pairs (single vectorized pass)...")

    # Compute the nth error pair to determine the threshold by their feature vector distance
    total_pairs_estimate = len(identities) * (len(identities) - 1) // 2
    target_rank = max(1, round((args.ppm / 1_000_000) * total_pairs_estimate))

    print(f"Total Pair Estimate: {total_pairs_estimate}, RANK PPM: {target_rank}")
    top_k_pairs, total_cross_pairs, all_pair_distances = find_top_k_impostor_pairs(
        embeddings, identities, filenames, model, k=max(target_rank, 8)
    )

    print(f"Total cross-identity pairs analyzed: {total_cross_pairs}")

    if target_rank > len(top_k_pairs):
        print(f"Error: target rank {target_rank} exceeds available pairs ({len(top_k_pairs)}).")
        return

    threshold = top_k_pairs[target_rank - 1]["distance"]
    achieved_ppm = round((target_rank / total_cross_pairs) * 1_000_000, 4)

    print(f"Total Cross Pairs: {total_cross_pairs}, Estimate: {total_pairs_estimate}")
    print("\n" + "=" * 55)
    print("                 THRESHOLD RESULTS")
    print("=" * 55)
    print(f"Model used             : {model}")
    print(f"Total cross pairs      : {total_cross_pairs}")
    print(f"Target ppm             : {args.ppm}")
    print(f"Rank used              : {target_rank}")
    print(f"Achieved ppm           : {achieved_ppm}")
    print(f"Calculated Threshold   : {threshold:.6f}")
    print("=" * 55)

    print(f"\nTop {min(8, len(top_k_pairs))} closest imposter pairs "
          f"(potential annotation errors if distances are very small):")
    for idx, pair in enumerate(top_k_pairs[:8]):
        print(f"  {idx + 1}. {pair['name_a']} ({pair['filename_a']}) <-> "
              f"{pair['name_b']} ({pair['filename_b']}) | Distance: {pair['distance']:.6f}")


if __name__ == "__main__":
    main()