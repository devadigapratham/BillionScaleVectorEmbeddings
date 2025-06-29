#!/usr/bin/env python3

import os
import sys
import pickle
import lzma
import time
import logging
import threading
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple, Generator
import hashlib
import json
import gc

import torch
import torch.multiprocessing as torch_mp
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [PID:%(process)d] - %(message)s',
    handlers=[
        logging.FileHandler('embedding_generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class CodeFile:
    path: str
    content: str
    size: int
    extension: str
    repo_name: str
    file_hash: str

@dataclass
class EmbeddingBatch:
    embeddings: np.ndarray
    metadata: List[Dict]
    batch_id: str

class CodeEmbeddingGenerator:
    def __init__(self, 
                 repos_dir: str = "./repositories",
                 output_dir: str = "./embeddings",
                 model_name: str = "microsoft/codebert-base",
                 max_length: int = 256,  # Reduced from 512
                 batch_size: int = 128,   
                 num_gpus: int = 4,
                 target_embeddings: int = 1_000_000_000):
        
        self.repos_dir = Path(repos_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.target_embeddings = target_embeddings
        
        # Code file extensions to process
        self.code_extensions = {
            '.c', '.cpp', '.cc', '.cxx', '.c++', '.h', '.hpp', '.hxx', '.h++',
            '.py', '.pyx', '.pxd', '.pyi',
            '.js', '.jsx', '.ts', '.tsx', '.mjs',
            '.java', '.scala', '.kt', '.kts',
            '.go', '.rs', '.php', '.rb', '.cs', '.fs', '.vb',
            '.swift', '.m', '.mm', '.pl', '.pm', '.r', '.R',
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
            '.sql', '.lua', '.jl', '.nim', '.zig', '.d', '.dart',
            '.html', '.css', '.scss', '.sass', '.less',
            '.xml', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.makefile', '.cmake', '.gradle', '.ant'
        }
        
        self.processed_files = 0
        self.generated_embeddings = 0
        self.failed_files = 0
        self.start_time = None
        
        logger.info(f"Initialized CodeEmbeddingGenerator with {self.num_gpus} GPUs")
        logger.info(f"Target embeddings: {self.target_embeddings:,}")
        logger.info(f"Max length: {self.max_length}, Batch size: {self.batch_size}")

    def find_code_files(self) -> Generator[CodeFile, None, None]:
        logger.info(f"Scanning repositories in {self.repos_dir}")
        
        for repo_path in self.repos_dir.iterdir():
            if not repo_path.is_dir():
                continue
                
            repo_name = repo_path.name
            logger.info(f"Processing repository: {repo_name}")
            
            for file_path in repo_path.rglob('*'):
                if not file_path.is_file():
                    continue
                    
                # Skip hidden files and directories
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                    
                extension = file_path.suffix.lower()
                if extension not in self.code_extensions:
                    continue
                
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Skip empty files or very small files
                    if len(content.strip()) < 10:
                        continue
                    
                    # Skip very large files (>500KB instead of 1MB)
                    if len(content) > 500_000:
                        continue
                    
                    # Create file hash for deduplication
                    file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                    
                    yield CodeFile(
                        path=str(file_path.relative_to(self.repos_dir)),
                        content=content,
                        size=len(content),
                        extension=extension,
                        repo_name=repo_name,
                        file_hash=file_hash
                    )
                    
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
                    continue

    def chunk_code_content(self, content: str, overlap: int = 25) -> List[str]:  # Reduced overlap
        lines = content.split('\n')
        chunks = []
        
        max_lines = self.max_length // 10  
        
        if len(lines) <= max_lines:
            return [content]
        
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line.split())  # Rough token count
            
            if current_length + line_length > max_lines and current_chunk:
                chunks.append('\n'.join(current_chunk))
                # Keep some overlap
                overlap_lines = min(overlap, len(current_chunk) // 3)  # Reduced overlap
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
                current_length = sum(len(l.split()) for l in current_chunk)
            
            current_chunk.append(line)
            current_length += line_length
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def get_memory_info(self, gpu_id: int):
        """Get GPU memory information"""
        try:
            torch.cuda.set_device(gpu_id)
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            return memory_allocated, memory_reserved, memory_total
        except:
            return 0, 0, 0

    def worker_process(self, gpu_id: int, file_queue: mp.Queue, result_queue: mp.Queue, 
                      stop_event, progress_queue: mp.Queue):
        try:
            # Set GPU and clear cache
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"Worker {gpu_id} starting on device {device}")

            # Load model with memory optimization
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use half precision
                low_cpu_mem_usage=True
            ).to(device)
            model.eval()

            # Enable gradient checkpointing to save memory
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

            batch_count = 0
            local_embeddings = 0
            last_repo_name = None

            # Process smaller micro-batches within each batch
            micro_batch_size = max(1, self.batch_size // 4)  # Process in smaller chunks
            
            while not stop_event.is_set():
                batch_files = []
                batch_chunks = []
                batch_metadata = []

                # Collect files for this batch
                for _ in range(self.batch_size):
                    try:
                        code_file = file_queue.get(timeout=1.0)
                        if code_file is None:
                            break

                        if last_repo_name != code_file.repo_name:
                            logger.info(f"GPU {gpu_id}: Creating embeddings from repo: {code_file.repo_name}")
                            last_repo_name = code_file.repo_name

                        chunks = self.chunk_code_content(code_file.content)
                        for i, chunk in enumerate(chunks):
                            batch_chunks.append(chunk)
                            batch_metadata.append({
                                'file_path': code_file.path,
                                'chunk_id': i,
                                'total_chunks': len(chunks),
                                'repo_name': code_file.repo_name,
                                'extension': code_file.extension,
                                'file_hash': code_file.file_hash,
                                'chunk_hash': hashlib.md5(chunk.encode('utf-8')).hexdigest()
                            })
                        batch_files.append(code_file)
                    except:
                        break

                if not batch_chunks:
                    continue

                try:
                    # Process in micro-batches to avoid OOM
                    all_embeddings = []
                    
                    for i in range(0, len(batch_chunks), micro_batch_size):
                        micro_chunks = batch_chunks[i:i + micro_batch_size]
                        
                        # Clear cache before each micro-batch
                        torch.cuda.empty_cache()
                        
                        inputs = tokenizer(
                            micro_chunks,
                            padding=True,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors='pt'
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs)
                            # Use mean pooling instead of just [CLS] token for better representations
                            attention_mask = inputs['attention_mask']
                            token_embeddings = outputs.last_hidden_state
                            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                            embeddings = embeddings.cpu().numpy().astype(np.float16)  # Convert to float16 to save memory
                            
                        all_embeddings.append(embeddings)
                        
                        # Clear intermediate tensors
                        del inputs, outputs, embeddings
                        torch.cuda.empty_cache()

                    # Concatenate all micro-batch embeddings
                    final_embeddings = np.concatenate(all_embeddings, axis=0)

                    batch_id = f"gpu_{gpu_id}_batch_{batch_count}"
                    embedding_batch = EmbeddingBatch(
                        embeddings=final_embeddings,
                        metadata=batch_metadata,
                        batch_id=batch_id
                    )

                    result_queue.put(embedding_batch)
                    batch_count += 1
                    local_embeddings += len(final_embeddings)
                    progress_queue.put(('embeddings', len(final_embeddings)))
                    progress_queue.put(('files', len(batch_files)))
                    
                    # Log memory usage periodically
                    if batch_count % 10 == 0:
                        mem_alloc, mem_reserved, mem_total = self.get_memory_info(gpu_id)
                        logger.info(f"GPU {gpu_id} batch {batch_count}: Memory {mem_alloc:.1f}GB/{mem_total:.1f}GB allocated")

                except Exception as e:
                    logger.error(f"GPU {gpu_id} error processing batch: {e}")
                    progress_queue.put(('errors', len(batch_files)))
                    # Clear cache on error
                    torch.cuda.empty_cache()
                    continue

            logger.info(f"Worker {gpu_id} completed. Total embeddings: {local_embeddings}")

        except Exception as e:
            logger.error(f"Worker {gpu_id} failed: {e}")
        finally:
            # Clean up
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            torch.cuda.empty_cache()
            gc.collect()

    def save_embeddings_batch(self, batch: EmbeddingBatch, batch_num: int):
        filename = self.output_dir / f"embeddings_batch_{batch_num:06d}.pkl.xz"
        
        try:
            data = {
                'embeddings': batch.embeddings,
                'metadata': batch.metadata,
                'batch_id': batch.batch_id,
                'timestamp': time.time(),
                'model_name': self.model_name
            }
            
            with lzma.open(filename, 'wb', preset=6) as f:
                pickle.dump(data, f)
            
            logger.debug(f"Saved batch {batch_num} with {len(batch.embeddings)} embeddings to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save batch {batch_num}: {e}")
            return False

    def progress_monitor(self, progress_queue, stop_event):
        stats = {
            'embeddings': 0,
            'files': 0,
            'errors': 0,
            'batches_saved': 0
        }
        
        last_update = time.time()
        
        while not stop_event.is_set():
            try:
                update_type, count = progress_queue.get(timeout=1.0)
                stats[update_type] += count
                
                if time.time() - last_update > 10:
                    elapsed = time.time() - self.start_time
                    rate = stats['embeddings'] / elapsed if elapsed > 0 else 0
                    
                    logger.info(f"Progress: {stats['embeddings']:,} embeddings, "
                              f"{stats['files']:,} files, {stats['errors']:,} errors, "
                              f"Rate: {rate:.1f} embeddings/sec")
                    
                    # Log GPU memory usage
                    for i in range(self.num_gpus):
                        mem_alloc, mem_reserved, mem_total = self.get_memory_info(i)
                        if mem_total > 0:
                            logger.info(f"GPU {i} Memory: {mem_alloc:.1f}GB/{mem_total:.1f}GB ({100*mem_alloc/mem_total:.1f}%)")
                    
                    last_update = time.time()
            except:
                continue

    def generate_embeddings(self):
        logger.info("Starting embedding generation process...")
        self.start_time = time.time()
        
        # Create queues with smaller sizes to avoid memory buildup
        file_queue = mp.Queue(maxsize=500)  # Reduced from 1000
        result_queue = mp.Queue(maxsize=50)  # Reduced from 100
        progress_queue = mp.Queue()
        stop_event = mp.Event()
        
        # Start worker processes
        workers = []
        for gpu_id in range(self.num_gpus):
            worker = mp.Process(
                target=self.worker_process,
                args=(gpu_id, file_queue, result_queue, stop_event, progress_queue)
            )
            worker.start()
            workers.append(worker)
        
        # Start progress monitor
        progress_thread = threading.Thread(
            target=self.progress_monitor,
            args=(progress_queue, stop_event)
        )
        progress_thread.start()
        
        # File producer thread
        def file_producer():
            try:
                file_count = 0
                for code_file in self.find_code_files():
                    file_queue.put(code_file)
                    file_count += 1
                    
                    if file_count * 3 > self.target_embeddings:
                        logger.info(f"Reached file limit for target embeddings: {file_count}")
                        break
                
                for _ in range(self.num_gpus):
                    file_queue.put(None)
                    
                logger.info(f"File producer finished. Total files queued: {file_count}")
                
            except Exception as e:
                logger.error(f"File producer error: {e}")
                stop_event.set()
        
        producer_thread = threading.Thread(target=file_producer)
        producer_thread.start()
        
        # Result consumer (main thread)
        batch_num = 0
        total_embeddings = 0
        
        try:
            while total_embeddings < self.target_embeddings:
                try:
                    batch = result_queue.get(timeout=30)
                    
                    if self.save_embeddings_batch(batch, batch_num):
                        batch_num += 1
                        total_embeddings += len(batch.embeddings)
                        progress_queue.put(('batches_saved', 1))
                        
                        logger.info(f"Saved batch {batch_num}, "
                                  f"Total embeddings: {total_embeddings:,}/{self.target_embeddings:,} "
                                  f"({100*total_embeddings/self.target_embeddings:.1f}%)")
                    
                except:
                    if not any(w.is_alive() for w in workers):
                        logger.info("All workers finished")
                        break
                    continue
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            logger.info("Shutting down...")
            stop_event.set()
            
            producer_thread.join(timeout=5)
            progress_thread.join(timeout=5)
            
            for worker in workers:
                worker.terminate()
                worker.join(timeout=5)
            
            elapsed = time.time() - self.start_time
            logger.info(f"Generation completed in {elapsed:.2f} seconds")
            logger.info(f"Total embeddings generated: {total_embeddings:,}")
            logger.info(f"Total batches saved: {batch_num}")
            logger.info(f"Average rate: {total_embeddings/elapsed:.1f} embeddings/sec")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate code embeddings using multiple GPUs')
    parser.add_argument('--repos-dir', default='./repositories', 
                        help='Directory containing cloned repositories')
    parser.add_argument('--output-dir', default='./embeddings',
                        help='Directory to save embedding files')
    parser.add_argument('--model-name', default='microsoft/codebert-base',
                        help='HuggingFace model name for embeddings')
    parser.add_argument('--batch-size', type=int, default=8,  # Reduced default
                        help='Batch size for processing')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs to use (default: 4)')
    parser.add_argument('--target-embeddings', type=int, default=1_000_000_000,
                        help='Target number of embeddings to generate (default: 1,000,000,000)')
    parser.add_argument('--max-length', type=int, default=256,  # Reduced default
                        help='Maximum sequence length for tokenization')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires GPU support.")
        sys.exit(1)

    available_gpus = torch.cuda.device_count()
    if available_gpus < args.num_gpus:
        logger.warning(f"Requested {args.num_gpus} GPUs but only {available_gpus} available")
        args.num_gpus = available_gpus

    logger.info(f"Using {args.num_gpus} GPUs for embedding generation")

    # Print GPU information
    for i in range(args.num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

    if not Path(args.repos_dir).exists():
        logger.error(f"Repositories directory does not exist: {args.repos_dir}")
        logger.error("Please run the repository cloning script first")
        sys.exit(1)

    generator = CodeEmbeddingGenerator(
        repos_dir=args.repos_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        target_embeddings=args.target_embeddings
    )

    try:
        generator.generate_embeddings()
        logger.info("Embedding generation completed successfully!")
        logger.info(f"Output files saved to: {args.output_dir}")
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        sys.exit(1)

def verify_embedding_files(output_dir: str):
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Output directory does not exist: {output_dir}")
        return

    embedding_files = list(output_path.glob("embeddings_batch_*.pkl.xz"))
    if not embedding_files:
        print(f"No embedding files found in {output_dir}")
        return

    print(f"\nFound {len(embedding_files)} embedding batch files")

    total_embeddings = 0
    total_size = 0
    sample_metadata = []

    for i, file_path in enumerate(sorted(embedding_files)[:5]):
        try:
            with lzma.open(file_path, 'rb') as f:
                data = pickle.load(f)

            embeddings = data['embeddings']
            metadata = data['metadata']

            total_embeddings += len(embeddings)
            total_size += file_path.stat().st_size

            if i == 0:
                sample_metadata = metadata[:3]

            print(f"Batch {i+1}: {len(embeddings)} embeddings, "
                  f"shape: {embeddings.shape}, "
                  f"size: {file_path.stat().st_size / (1024*1024):.1f} MB")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if embedding_files:
        avg_embeddings_per_file = total_embeddings / min(5, len(embedding_files))
        estimated_total = avg_embeddings_per_file * len(embedding_files)

        avg_size_per_file = total_size / min(5, len(embedding_files))
        estimated_total_size = avg_size_per_file * len(embedding_files)

        print(f"\nEstimated totals:")
        print(f"Total embeddings: ~{estimated_total:,.0f}")
        print(f"Total compressed size: ~{estimated_total_size / (1024*1024*1024):.1f} GB")

    if sample_metadata:
        print(f"\nSample metadata:")
        for meta in sample_metadata:
            print(f"  - {meta['repo_name']}/{meta['file_path']} "
                  f"(chunk {meta['chunk_id']+1}/{meta['total_chunks']})")

if __name__ == "__main__":
    if hasattr(torch.multiprocessing, 'set_start_method'):
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        output_dir = sys.argv[2] if len(sys.argv) > 2 else './embeddings'
        verify_embedding_files(output_dir)
    else:
        main()