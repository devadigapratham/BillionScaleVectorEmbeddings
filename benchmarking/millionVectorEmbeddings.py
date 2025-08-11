import os
import pickle
import lzma
import logging
import argparse
from pathlib import Path
import numpy as np

OUTPUT_DIR = "./random_embeddings"
BATCH_SIZE = 10000  # Shard size for saving
NUM_VECTORS = 1000000
DIMS = [128, 256, 512, 1024, 2048]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class RandomEmbedder:
    def __init__(self):
        self.output_base = Path(OUTPUT_DIR)
        self.output_base.mkdir(exist_ok=True)

    def _save_shard(self, embeddings: np.ndarray, dim: int, shard_idx: int):
        dim_dir = self.output_base / f"dim_{dim}"
        dim_dir.mkdir(exist_ok=True)
        out_path = dim_dir / f"batch_{shard_idx:06d}.pkl.xz"
        with lzma.open(out_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"Saved {len(embeddings)} embeddings for dim {dim} to {out_path}")

    def run(self):
        for dim in DIMS:
            logger.info(f"Generating {NUM_VECTORS:,} random embeddings for dimension {dim}")
            total_generated = 0
            shard_idx = 0

            while total_generated < NUM_VECTORS:
                batch_size = min(BATCH_SIZE, NUM_VECTORS - total_generated)
                # Generate random vectors from standard normal distribution
                embeddings = np.random.randn(batch_size, dim).astype(np.float16)
                self._save_shard(embeddings, dim, shard_idx)
                total_generated += batch_size
                shard_idx += 1

            logger.info(f"Completed generation for dimension {dim}. Total embeddings: {total_generated:,}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate random vector embeddings")
    parser.add_argument("--num-vectors", type=int, default=NUM_VECTORS, help="Number of vectors to generate (default: 1,000,000)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for sharding (default: 10,000)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    globals()["NUM_VECTORS"] = args.num_vectors
    globals()["BATCH_SIZE"] = args.batch_size
    
    embedder = RandomEmbedder()
    embedder.run()