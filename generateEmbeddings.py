import os
import pickle
import lzma
import logging
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Generator

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

REPOS_ROOT = "./repositories"
OUTPUT_DIR = "./embeddings_billion"
MODEL_NAME = "microsoft/codebert-base"
MAX_TOKEN_LENGTH = 512
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
    def __init__(self):
        self.repos_root = Path(REPOS_ROOT)
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self.device = torch.device("cuda:0")
        logger.info(f"Using device: {torch.cuda.get_device_name(0)}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(self.device)
        self.model.eval()

        self.progress_path = self.output_dir / "progress.log"
        self.global_idx = 0
        if self.progress_path.exists():
            with open(self.progress_path) as f:
                self.global_idx = int(f.read().strip())

    def _find_code_files(self) -> Generator[CodeFile, None, None]:
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
        out_path = self.output_dir / f"batch_{self.global_idx:012d}.pkl.xz"
        with lzma.open(out_path, 'wb') as f:
            pickle.dump(embeddings.astype(np.float16), f)
        logger.info(f"Saved {len(embeddings)} embeddings to {out_path}")
        self.global_idx += 1
        with open(self.progress_path, 'w') as f:
            f.write(str(self.global_idx))

    def run(self):
        total_embeddings = self.global_idx * BATCH_SIZE
        embeddings_buffer = []
        chunks_to_process = []

        logger.info("Scanning all repositories...")
        #files = list(self._find_code_files())
        #logger.info(f"Found {len(files)} files across all repositories")

        file_counter = 0
        start_time = time.time()

        for code_file in (self._find_code_files()):
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
                        throughput = total_embeddings / elapsed
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

if __name__ == "__main__":
    CodeEmbedder().run()
