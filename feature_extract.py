import os
import argparse
import numpy as np
import cv2
from models import MobileNetV2CNN, KerasFaceNet, ArcFace, SFace, FaceAligner
from loader import DB_PATHS, load_ldb2
 
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
    parser.add_argument('--output', default="face_db.npy", help="Path to save the .npy database.")

    args = parser.parse_args()
    args.output = args.output.lower()

    if not args.output.endswith('.npy'):
        raise ValueError("--output should end with .npy extension")
    
    print(args)
    build_database(args.db, args.model, args.output)

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

def build_database(db, model_name, output="face_db.npy"):
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
            continue
        
        entry = feature_db.setdefault(person, {"embeddings": [], "filenames": []})
        entry['embeddings'].append(np.asarray(embedding, dtype=np.float32))
        entry['filenames'].append(os.path.basename(image_path))
        
    for person, entry in feature_db.items():
        entry['embeddings'] = np.stack(entry['embeddings'])
    
    np.save(f"features/{output}", feature_db, allow_pickle=True)
    
    print(f"\nSaved {len(feature_db)} identities -> {output}")
    print(f"Unreadable files: {len(skipped)}")
    print(f"No face detected: {len(no_faces)}")

    return feature_db

    
if __name__ == "__main__":
    main()
