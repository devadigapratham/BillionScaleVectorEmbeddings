"""
Codebase Embedding Generator

This script extracts code snippets from the GCC compiler codebase (customizable in the code) and generates
semantic embeddings using transformer models. It processes code in parallel batches
using Modal's cloud GPU infrastructure to create vector representations and compresses it to pickle+lzma format for efficient storage and retrieval.
"""

import modal
import os
import pickle
import lzma
import time
from pathlib import Path
from typing import List, Tuple
import threading
import queue
import logging
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'embedding_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Modal setup
app = modal.App("embedding-generator")

# Create Modal image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "sentence-transformers",
    "torch",
    "transformers",
    "numpy",
    "scikit-learn"
])

# Modal volume for storing results
volume = modal.Volume.from_name("embedding-results", create_if_missing=True)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=2, size="40GB"),
    volumes={"/results": volume},
    timeout=3600  # 1 hour timeout
)
def generate_embeddings_batch(
    batch_texts: List[str], 
    batch_id: int, 
    model_name: str,
    embedding_dim: int
):
    """Generate embeddings for a batch of texts using specified model"""
    import torch
    from sentence_transformers import SentenceTransformer
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Batch {batch_id}: Using device: {device}")
    logger.info(f"Batch {batch_id}: CUDA devices available: {torch.cuda.device_count()}")
    
    start_time = time.time()
    
    try:
        # Load model
        logger.info(f"Batch {batch_id}: Loading model {model_name}")
        model = SentenceTransformer(model_name, device=device)
        
        # Generate embeddings
        logger.info(f"Batch {batch_id}: Generating embeddings for {len(batch_texts)} texts")
        embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=True)
        
        # Prepare data for saving
        batch_data = {
            'embeddings': embeddings,
            'texts': batch_texts,
            'model_name': model_name,
            'embedding_dim': embedding_dim,
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'processing_time': time.time() - start_time
        }
        
        # Save to pickle_lzma format
        filename = f"/results/embeddings_batch_{batch_id}_{model_name.replace('/', '_')}.pkl.xz"
        with lzma.open(filename, 'wb') as f:
            pickle.dump(batch_data, f)
        
        logger.info(f"Batch {batch_id}: Successfully saved {len(embeddings)} embeddings to {filename}")
        logger.info(f"Batch {batch_id}: Processing time: {time.time() - start_time:.2f} seconds")
        
        # Commit volume changes
        volume.commit()
        
        return {
            'batch_id': batch_id,
            'num_embeddings': len(embeddings),
            'filename': filename,
            'processing_time': time.time() - start_time,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Batch {batch_id}: Error generating embeddings: {str(e)}")
        return {
            'batch_id': batch_id,
            'error': str(e),
            'success': False
        }

def extract_code_tokens(codebase_path: str, max_tokens: int = 100000) -> List[str]:
    """Extract code snippets from the GCC codebase"""
    logger.info(f"Extracting code tokens from {codebase_path}")
    
    code_snippets = []
    token_count = 0
    
    # File extensions to include
    code_extensions = {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.py', '.java', '.js', '.ts'}
    
    for root, dirs, files in os.walk(codebase_path):
        for file in files:
            if any(file.endswith(ext) for ext in code_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # Split into chunks (functions, classes, or logical blocks)
                        lines = content.split('\n')
                        current_chunk = []
                        
                        for line in lines:
                            current_chunk.append(line)
                            
                            # Create chunks of ~50-100 lines or on function boundaries
                            if (len(current_chunk) >= 50 and 
                                (line.strip().startswith('}') or line.strip() == '' or 
                                 line.strip().startswith('function') or line.strip().startswith('def'))):
                                
                                chunk_text = '\n'.join(current_chunk).strip()
                                if len(chunk_text) > 50:  # Minimum chunk size
                                    code_snippets.append(chunk_text)
                                    token_count += len(chunk_text.split())
                                    
                                    if token_count >= max_tokens:
                                        logger.info(f"Reached maximum token limit: {token_count}")
                                        return code_snippets
                                
                                current_chunk = []
                        
                        # Add remaining chunk if any
                        if current_chunk:
                            chunk_text = '\n'.join(current_chunk).strip()
                            if len(chunk_text) > 50:
                                code_snippets.append(chunk_text)
                                token_count += len(chunk_text.split())
                
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {str(e)}")
                    continue
                
                if token_count >= max_tokens:
                    break
        
        if token_count >= max_tokens:
            break
    
    logger.info(f"Extracted {len(code_snippets)} code snippets with ~{token_count} tokens")
    return code_snippets

@app.local_entrypoint()
def main():
    """Main function to orchestrate embedding generation"""
    logger.info("=== Starting Embedding Generation Process ===")
    
    # Configuration
    GCC_PATH = "./embedding_datasets/gcc"  # Adjust path as needed
    MAX_TOKENS = 100000
    BATCH_SIZE = 25  # Adjust based on memory constraints
    
    # Models to use
    models = [
        ("sentence-transformers/all-MiniLM-L6-v2", 384),  # MiniLM for 384 dims
        ("sentence-transformers/all-mpnet-base-v2", 768)   # MPNet for 768 dims
    ]
    
    # Extract code snippets
    logger.info("Phase 1: Extracting code snippets from GCC codebase")
    code_snippets = extract_code_tokens(GCC_PATH, MAX_TOKENS)
    
    if not code_snippets:
        logger.error("No code snippets extracted! Check the GCC path.")
        return
    
    logger.info(f"Total code snippets extracted: {len(code_snippets)}")
    
    # Process each model
    for model_name, embedding_dim in models:
        logger.info(f"\n=== Processing model: {model_name} (dim: {embedding_dim}) ===")
        
        # Create batches
        batches = [code_snippets[i:i + BATCH_SIZE] for i in range(0, len(code_snippets), BATCH_SIZE)]
        logger.info(f"Created {len(batches)} batches of size ~{BATCH_SIZE}")
        
        # Process batches in parallel using Modal
        logger.info("Phase 2: Generating embeddings in parallel batches")
        
        # Submit all batch jobs
        batch_jobs = []
        for batch_id, batch_texts in enumerate(batches):
            logger.info(f"Submitting batch {batch_id + 1}/{len(batches)} with {len(batch_texts)} snippets")
            job = generate_embeddings_batch.spawn(
                batch_texts, 
                batch_id, 
                model_name,
                embedding_dim
            )
            batch_jobs.append(job)
        
        # Collect results
        results = []
        successful_batches = 0
        total_embeddings = 0
        
        for i, job in enumerate(batch_jobs):
            try:
                result = job.get()
                results.append(result)
                
                if result['success']:
                    successful_batches += 1
                    total_embeddings += result['num_embeddings']
                    logger.info(f"Batch {result['batch_id']} completed: {result['num_embeddings']} embeddings in {result['processing_time']:.2f}s")
                else:
                    logger.error(f"Batch {i} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Failed to get result for batch {i}: {str(e)}")
        
        # Summary for this model
        logger.info(f"\n=== Summary for {model_name} ===")
        logger.info(f"Total batches: {len(batches)}")
        logger.info(f"Successful batches: {successful_batches}")
        logger.info(f"Total embeddings generated: {total_embeddings}")
        logger.info(f"Success rate: {successful_batches/len(batches)*100:.1f}%")
    
    logger.info("\n=== Embedding Generation Complete ===")
    logger.info("Files saved in Modal volume 'embedding-results'")
    logger.info("Use Modal dashboard or CLI to download results")
    

if __name__ == "__main__":
    main()