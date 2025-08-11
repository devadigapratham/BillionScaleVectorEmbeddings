import os
import pickle
import lzma
from pathlib import Path
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "./random_embeddings"
DIMS = [128, 256, 512, 1024, 2048]
EXPECTED_VECTORS = 1000000

def verify_embeddings():
    output_base = Path(OUTPUT_DIR)
    summary = []
    
    for dim in DIMS:
        dim_dir = output_base / f"dim_{dim}"
        total_vectors = 0
        total_size_bytes = 0
        file_count = 0
        errors = []
        
        if not dim_dir.exists():
            errors.append(f"Directory for dimension {dim} does not exist")
            summary.append((dim, total_vectors, file_count, total_size_bytes, errors))
            continue
            
        # Iterate through all .pkl.xz files in dimension directory
        for file_path in dim_dir.glob("batch_*.pkl.xz"):
            try:
                with lzma.open(file_path, 'rb') as f:
                    embeddings = pickle.load(f)
                
                if not isinstance(embeddings, np.ndarray):
                    errors.append(f"File {file_path} does not contain a numpy array")
                    continue
                    
                num_vectors, embedding_dim = embeddings.shape
                if embedding_dim != dim:
                    errors.append(f"File {file_path} has incorrect dimension: {embedding_dim} (expected {dim})")
                
                total_vectors += num_vectors
                file_size = file_path.stat().st_size
                total_size_bytes += file_size
                file_count += 1
                
            except Exception as e:
                errors.append(f"Error processing {file_path}: {str(e)}")
        
        summary.append((dim, total_vectors, file_count, total_size_bytes, errors))
    
    # Print summary at the end
    logger.info("Verification Summary:")
    logger.info("=" * 50)
    for dim, total_vectors, file_count, total_size_bytes, errors in summary:
        # Calculate size in human-readable format
        if total_size_bytes < 1024:
            size_str = f"{total_size_bytes} bytes"
        elif total_size_bytes < 1024 * 1024:
            size_str = f"{total_size_bytes / 1024:.2f} KB"
        elif total_size_bytes < 1024 * 1024 * 1024:
            size_str = f"{total_size_bytes / (1024 * 1024):.2f} MB"
        else:
            size_str = f"{total_size_bytes / (1024 * 1024 * 1024):.2f} GB"
        
        status = "✓" if total_vectors == EXPECTED_VECTORS else "✗"
        logger.info(f"Dimension {dim}:")
        logger.info(f"  Total vectors: {total_vectors:,} {status}")
        logger.info(f"  Expected vectors: {EXPECTED_VECTORS:,}")
        logger.info(f"  Number of files: {file_count}")
        logger.info(f"  Total size: {size_str}")
        if errors:
            logger.info("  Errors:")
            for error in errors:
                logger.info(f"    - {error}")
        logger.info("-" * 50)

if __name__ == "__main__":
    verify_embeddings()