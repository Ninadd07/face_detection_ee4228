import os
import glob
import numpy as np

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

def load_person_embeddings(emb_root="embeddings", min_per_person=5):
    """
    Loads all .npy files under embeddings/<person_name>/ into a dict:
    {
      "jessica": np.ndarray of shape (N, 512),
      ...
    }
    """
    person_to_embs = {}

    if not os.path.isdir(emb_root):
        print(f"[ERROR] embeddings root '{emb_root}' not found")
        return person_to_embs

    for person_name in sorted(os.listdir(emb_root)):
        person_dir = os.path.join(emb_root, person_name)
        if not os.path.isdir(person_dir):
            continue

        all_embs = []
        for path in glob.glob(os.path.join(person_dir, "*.npy")):
            embs = np.load(path)
            if embs.ndim == 1:
                embs = embs[None, :]
            all_embs.append(embs)

        if not all_embs:
            continue

        all_embs = np.vstack(all_embs)
        if all_embs.shape[0] < min_per_person:
            print(f"[WARN] {person_name} has only {all_embs.shape[0]} embeddings")
        person_to_embs[person_name] = all_embs

    print(f"[INFO] Loaded embeddings for persons: {list(person_to_embs.keys())}")
    return person_to_embs

def predict_person(embedding, person_to_embs, threshold=0.5, metric="cosine"):
    """
    Given one embedding and dict of stored embeddings, return (best_name, best_score)
    or ("unknown", best_score) if below threshold.
    """
    best_name = "unknown"
    best_score = -1.0 if metric == "cosine" else 1e9

    for name, embs in person_to_embs.items():
        if metric == "cosine":
            sims = np.dot(embs, embedding) / (
                np.linalg.norm(embs, axis=1) * np.linalg.norm(embedding) + 1e-8
            )
            score = float(np.mean(sims))  # average similarity to this person
            if score > best_score:
                best_score = score
                best_name = name
        else:  # L2 distance
            dists = np.linalg.norm(embs - embedding, axis=1)
            score = float(np.mean(dists))
            if score < best_score:
                best_score = score
                best_name = name

    # thresholding for cosine (you can tune this)
    if metric == "cosine":
        if best_score < threshold:
            return "unknown", best_score
    else:
        if best_score > threshold:
            return "unknown", best_score

    return best_name, best_score
