import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv

"""
NOTE: Change the paths to the database paths you have set up in your directory. Use absolute pathing.
"""
DB_PATHS = {
    "LaSalleDB1": "D:/FILES/PROGRAMMING/face-detection/LaSalleDB1/original",
    "LaSalleDB2": "D:/FILES/PROGRAMMING/face-detection/LaSalleDB1",  # light/medium/heavy variants
    "LFW": "D:/FILES/PROGRAMMING/face-detection/lfw/images",
}

def plot(independent, threshold=1.128,
         model='',
         output_path='results/images/frequency_distance_plot.png', show=True):
    """
        KDE Chart Visualization of Independence Test performed per model.
    """
    title = f"Frequency vs Distance ({model} @ Threshold ({threshold}))"
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(8, 5.5))
 
    sns.histplot(independent, bins=30, kde=True, color='red', alpha=0.3)
    plt.axvline(x=threshold, color='blue', linestyle='--', label=f'Threshold ({threshold})')
    plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()
 
 
def plot_fp_images(pairs_fp, gallery, distance_key='distance',
                    title="Top 3 False Positive Pairs\n(Lowest Discriminative Ability)",
                    output_path='fp_top3.png', show=True):
    """
    Same idea as your original: show the 3 impostor pairs the model found
    hardest to tell apart. Two generalizations:
 
    1. Sorts by a single `distance_key` instead of (l2, -cosine_sim), since
       only SFace naturally produces both metrics -- MobileNetV2/FaceNet
       only have an L2 distance, ArcFace only has a cosine distance. All
       four models agree on "lower distance = more similar", so sorting by
       that one shared quantity works uniformly.
    2. Looks up each identity's actual gallery image via
       gallery[identity]["gallery_path"] (saved by build_features_multimodel.py)
       instead of assuming the LFW-only f"{name}_0001.jpg" naming pattern --
       so this now works for LaSalleDB1/LaSalleDB2 too, not just LFW.
    """
    sorted_fp = sorted(pairs_fp, key=lambda x: x[distance_key])
    top_3_fp = sorted_fp[:3]
 
    if not top_3_fp:
        print("No false positive pairs to plot.")
        return
 
    fig, axes = plt.subplots(len(top_3_fp), 2, figsize=(10, 8), squeeze=False)
    fig.suptitle(title, fontsize=16)
 
    for i, pair in enumerate(top_3_fp):
        p1, p2 = pair['name1'], pair['name2']
        score = pair[distance_key]
 
        img1_path = gallery[p1]["gallery_path"]
        img2_path = gallery[p2]["gallery_path"]
 
        img1 = cv.cvtColor(cv.imread(img1_path), cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(cv.imread(img2_path), cv.COLOR_BGR2RGB)
 
        axes[i, 0].imshow(img1)
        axes[i, 0].set_title(f"Pair {i+1}A: {p1.replace('_', ' ')}")
        axes[i, 0].axis('off')
 
        axes[i, 1].imshow(img2)
        axes[i, 1].set_title(f"Pair {i+1}B: {p2.replace('_', ' ')}\nDistance: {score}")
        axes[i, 1].axis('off')
 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()
 
 
def load_ldb2():
    """
    Unchanged from your version, just reads DB_PATHS["LaSalleDB2"] instead
    of a separately hardcoded BASE_DIR, so there's one source of truth for
    that path across the whole project.
    """
    CONDITIONS = ['light', 'heavy', 'medium']
    BASE_DIR = DB_PATHS["LaSalleDB2"]
    db = []
 
    for condition in CONDITIONS:
        condition_path = os.path.join(BASE_DIR, condition)
        if not os.path.isdir(condition_path):
            continue
 
        for person_name in os.listdir(condition_path):
            person_path = os.path.join(condition_path, person_name)
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_file)
                    db.append((person_name, img_path))
 
    return db
 
 
if __name__ == "__main__":
    db = load_ldb2()
    print(f"Total images: {len(db)}")