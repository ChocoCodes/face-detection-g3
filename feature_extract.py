import os
import argparse
import numpy as np
import cv2
from models import MobileNetV2CNN, KerasFaceNet, ArcFace, SFace, FaceAligner
from loader import DB_PATHS, load_ldb2
import json

MODELS = {
    'mobilenet' : MobileNetV2CNN,
    'facenet' : KerasFaceNet,
    'arcface' : ArcFace,
    'sface' : SFace
}

IMG_EXT = ['.jpg', '.jpeg']

def main():
    parser = argparse.ArgumentParser(description="Build a single-model vector face database.")

    parser.add_argument('--db', required=True, choices=list(DB_PATHS), help="Choose between the ff database: LaSalleDB1, LaSalleDB2, LFW, LFW2 (Coming Soon).")
    parser.add_argument('--model', required=True, choices=list(MODELS), help="Which model will be used for building the feature database.")

    args = parser.parse_args()
    db = args.db.lower()
    output = f"{db}-{args.model}.npy"
    
    print(output)
    build_database(args.db, args.model, output)

def load_database(path):
    if path is None:
        raise ValueError("Path is not specified.")
    return np.load(path, allow_pickle=True).item()

def normalize_brightness(bgr_face):
    lab = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def build_database(db, model_name, output):
    print(f"Loading {model_name} model...")
    aligner = FaceAligner()
    model = MODELS[model_name]()

    if db == "LaSalleDB2":
        pairs = load_ldb2()
    else: 
        image_db = DB_PATHS[db]
        pairs = [
            (person_name, os.path.join(image_db, person_name, fname))
            for person_name in sorted(os.listdir(image_db))
            if os.path.isdir(os.path.join(image_db, person_name))
            for fname in sorted(os.listdir(os.path.join(image_db, person_name)))
            if fname.lower().endswith(tuple(IMG_EXT))
        ]

    feature_db = {}
    skipped = []
    no_faces = []
    embedding_failed = []

    for person, image_path in pairs:
        if not image_path.lower().endswith(tuple(IMG_EXT)):
            continue
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"  [skip] unreadable: {image_path}")
            skipped.append(image_path)
            continue
        
        aligned_face = aligner.align(img)
        if aligned_face is None:
            print(f"  [skip] no face detected: {image_path}")
            no_faces.append(image_path)
            continue
        
        aligned_face = normalize_brightness(aligned_face)
        try:
            embedding = model.get_embedding(aligned_face)
        except Exception as e:
            print(f"  [skip] embedding failed for {image_path}: {e}")
            embedding_failed.append(image_path)
            continue
        
        entry = feature_db.setdefault(person, {"embeddings": [], "filenames": []})
        entry['embeddings'].append(np.asarray(embedding, dtype=np.float32))
        entry['filenames'].append(os.path.basename(image_path))
        
    for person, entry in feature_db.items():
        entry['embeddings'] = np.stack(entry['embeddings'])
    
    np.save(f"features/{output}", feature_db, allow_pickle=True)
    
    print(f"\nSaved {len(feature_db)} identities -> {output}")

    n_skipped = len(skipped)
    n_no_faces = len(no_faces)
    n_embedding_failed = len(embedding_failed)
    print(f"Unreadable files: {n_skipped} | No face detected: {n_no_faces}")
    if n_skipped > 0 or n_no_faces > 0 or n_embedding_failed > 0:
        log_name = output.replace(".npy",".json")
        with open(log_name, "w") as f:
            json.dump({
                "model" : model_name,
                "database" : db,
                "unreadable_images" : skipped,
                "n_unreadable" : n_skipped,
                "images_no_faces" : no_faces,
                "n_no_faces" : n_no_faces,
                "embedding_failed" : embedding_failed,
                "n_embedding_failed" : n_embedding_failed
            }, f, indent=4)        
    
    return feature_db

    
if __name__ == "__main__":
    main()
