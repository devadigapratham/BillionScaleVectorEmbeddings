import os
import pickle
import lzma
import logging
import hashlib
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Generator

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

REPOS_ROOT = "./repositories"
OUTPUT_DIR = "./embeddings_billion"
MODEL_NAME = "microsoft/codebert-base"
MAX_TOKEN_LENGTH = 128
BATCH_SIZE = 2048
TARGET_EMBEDDING_COUNT = 1_000_000_000

CHUNK_LINES = 8
STRIDE_LINES = 4

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class CodeFile:
    path: str
    content: str
    repo_name: str
    file_hash: str

class CodeEmbedder:
    def __init__(self, gpu_id=0, index_file_path=None, start_index=None, end_index=None):
        self.repos_root = Path(REPOS_ROOT)
        self.output_dir = Path(OUTPUT_DIR)
        
        self.gpu_id = gpu_id
        self.index_file_path = index_file_path
        self.start_index = start_index
        self.end_index = end_index

        self.output_dir.mkdir(exist_ok=True)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self.device = torch.device(f"cuda:{self.gpu_id}")
        logger.info(f"Using device: {torch.cuda.get_device_name(self.gpu_id)}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(self.device)
        self.model.eval()

        self.global_idx = 0 
        self.progress_path = self.output_dir / "progress.log"
        if self.progress_path.exists():
            self.progress_path.unlink()

    def _read_files_from_index(self) -> Generator[CodeFile, None, None]:
        if not self.index_file_path or not Path(self.index_file_path).exists():
            raise FileNotFoundError(f"Index file not found: {self.index_file_path}")
        
        logger.info(f"Reading files from index: {self.index_file_path}")
        logger.info(f"Processing range: {self.start_index} to {self.end_index}")
        
        with open(self.index_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' not in line:
                    continue
                
                try:
                    index_str, file_path = line.split(':', 1)
                    index_num = int(index_str.strip())
                    file_path = file_path.strip()
                    
                    if self.start_index is not None and index_num < self.start_index:
                        continue
                    if self.end_index is not None and index_num > self.end_index:
                        continue
                    
                    path_obj = Path(file_path)
                    if not path_obj.exists() or not path_obj.is_file():
                        logger.warning(f"File not found: {file_path}")
                        continue
                    
                    try:
                        content = path_obj.read_text(encoding="utf-8", errors="ignore")
                        if not (10 < len(content) < 500_000): 
                            continue
                        if '\x00' in content:
                            continue
                        
                        repo_name = path_obj.parts[0] if len(path_obj.parts) > 0 else "unknown"
                        
                        yield CodeFile(
                            path=str(path_obj),
                            content=content,
                            repo_name=repo_name,
                            file_hash=hashlib.md5(content.encode()).hexdigest()
                        )
                    except Exception as e:
                        logger.warning(f"Skipping {file_path}: {e}")
                        
                except ValueError as e:
                    logger.warning(f"Invalid line format: {line}")
                    continue

    def _find_code_files(self) -> Generator[CodeFile, None, None]:
        if self.index_file_path:
            yield from self._read_files_from_index()
            return
            
        for repo_dir in sorted(self.repos_root.iterdir()):
            if not repo_dir.is_dir() or repo_dir.name.startswith(".git") or repo_dir.name.startswith(".github"):
                continue
            repo_name = repo_dir.name
            for file_path in repo_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if any(part.startswith(".git") or part.startswith(".github") for part in file_path.parts):
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    if not (10 < len(content) < 500_000): 
                        continue
                    if '\x00' in content:
                        continue
                    yield CodeFile(
                        path=str(file_path.relative_to(self.repos_root)),
                        content=content,
                        repo_name=repo_name,
                        file_hash=hashlib.md5(content.encode()).hexdigest()
                    )
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {e}")

    def _chunk_content(self, content: str) -> List[str]:
        lines = content.split('\n')
        chunks = []
        for i in range(0, len(lines) - CHUNK_LINES + 1, STRIDE_LINES):
            chunk = lines[i:i + CHUNK_LINES]
            chunks.append('\n'.join(chunk))
        if len(lines) < CHUNK_LINES:
            chunks.append('\n'.join(lines))
        return chunks

    def _save_shard(self, embeddings: np.ndarray):
        out_path = self.output_dir / f"gpu{self.gpu_id}_batch_{self.global_idx:012d}.pkl.xz"
        with lzma.open(out_path, 'wb') as f:
            pickle.dump(embeddings.astype(np.float16), f)
        logger.info(f"Saved {len(embeddings)} embeddings to {out_path}")
        self.global_idx += 1

    def run(self):
        total_embeddings = 0
        embeddings_buffer = []
        chunks_to_process = []

        logger.info("Scanning all repositories...")
        file_counter = 0
        start_time = time.time()

        for code_file in self._find_code_files():
            file_counter += 1
            chunks = self._chunk_content(code_file.content)

            for chunk in chunks:
                if total_embeddings >= TARGET_EMBEDDING_COUNT:
                    logger.info("Target reached.")
                    return

                chunks_to_process.append(chunk)

                if len(chunks_to_process) >= BATCH_SIZE:
                    try:
                        inputs = self.tokenizer(chunks_to_process, padding=True, truncation=True,
                                                max_length=MAX_TOKEN_LENGTH, return_tensors='pt')
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            output = self.model(**inputs).last_hidden_state[:, 0, :]

                        embeddings = output.cpu().numpy().astype(np.float16)
                        embeddings_buffer.extend(embeddings)
                        total_embeddings += len(embeddings)

                        elapsed = time.time() - start_time
                        progress_pct = (total_embeddings / TARGET_EMBEDDING_COUNT) * 100
                        throughput = total_embeddings / max(elapsed, 1)
                        eta = (TARGET_EMBEDDING_COUNT - total_embeddings) / throughput
                        logger.info(f"Progress: {progress_pct:.4f}% | Total embeddings so far: {total_embeddings:,} | ETA: {eta/3600:.2f} hours | Files processed: {file_counter} | Throughput: {throughput:.2f} embeddings/sec")

                    except Exception as e:
                        logger.error(f"Model error: {e}")
                    finally:
                        chunks_to_process.clear()

                    while len(embeddings_buffer) >= BATCH_SIZE:
                        to_save = embeddings_buffer[:BATCH_SIZE]
                        self._save_shard(np.array(to_save))
                        embeddings_buffer = embeddings_buffer[BATCH_SIZE:]

        if embeddings_buffer:
            self._save_shard(np.array(embeddings_buffer))

        logger.info(f"Done. Total embeddings: {total_embeddings:,}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate code embeddings")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--index-file", type=str, help="Path to the index file (file_index.txt)")
    parser.add_argument("--start-index", type=int, help="Start index for processing files")
    parser.add_argument("--end-index", type=int, help="End index for processing files")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for processing (default: 2048)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.batch_size is not None:
        globals()["BATCH_SIZE"] = args.batch_size
        
    if args.index_file and (args.start_index is None or args.end_index is None):
        logger.warning("When using --index-file, both --start-index and --end-index should be specified")
    
    embedder = CodeEmbedder(
        gpu_id=args.gpu_id,
        index_file_path=args.index_file,
        start_index=args.start_index,
        end_index=args.end_index
    )
    embedder.run()