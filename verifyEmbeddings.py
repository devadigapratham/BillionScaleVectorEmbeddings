import os
import lzma
import pickle
from pathlib import Path

EMBEDDINGS_DIR = "./embeddings_billion" 

def verify_embeddings(directory: str):
    total_files = 0
    total_embeddings = 0
    expected_dim = None

    for file in sorted(Path(directory).glob("batch_*.pkl.xz")):
        try:
            with lzma.open(file, "rb") as f:
                data = pickle.load(f)

            embeddings = data.get("embeddings")
            metadata = data.get("metadata")
            model_name = data.get("model_name")

            if embeddings is None or metadata is None:
                print(f"[ERROR] {file.name} missing data keys.")
                continue

            num_embeds, dim = embeddings.shape
            if expected_dim is None:
                expected_dim = dim
            elif dim != expected_dim:
                print(f"[WARN] {file.name} has unexpected dim {dim}, expected {expected_dim}.")

            if num_embeds != len(metadata):
                print(f"[MISMATCH] {file.name}: {num_embeds} embeddings but {len(metadata)} metadata entries.")

            print(f"[OK] {file.name}: {num_embeds} embeddings, dim={dim}")
            total_files += 1
            total_embeddings += num_embeds

        except Exception as e:
            print(f"[FAIL] {file.name}: {e}")

    print("\nVerification complete.")
    print(f"Total files: {total_files}")
    print(f"Total embeddings: {total_embeddings}")
    print(f"Expected embedding dimension: {expected_dim}")

if __name__ == "__main__":
    verify_embeddings(EMBEDDINGS_DIR)
