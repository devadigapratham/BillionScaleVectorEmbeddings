#!/usr/bin/env python3

import os
import subprocess
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('repo_cloning.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RepoCloner:
    """
    A class to clone a list of Git repositories in parallel,
    update them if they already exist, collect statistics, and
    create a global index of all files.
    """
    def __init__(self, base_dir="./repositories", max_workers=8, index_filename="file_index.txt"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        
        # --- State and Locking ---
        self.cloned_repos = []
        self.failed_repos = []
        self.file_index_counter = 0
        self.index_lock = threading.Lock() # For thread-safe access to the index counter and file

        # --- File Indexing Setup ---
        self.index_file_path = Path(index_filename)
        # Clear the index file at the start of a new run
        if self.index_file_path.exists():
            self.index_file_path.unlink()
        logger.info(f"File index will be written to: {self.index_file_path.absolute()}")

        # --- REPOSITORY LIST (UPDATED) ---
        self.repositories = sorted(list(set([
            # Linux Kernel and Core Systems
            ("https://github.com/torvalds/linux.git", "linux"),
            ("https://github.com/systemd/systemd.git", "systemd"),
            ("https://github.com/util-linux/util-linux.git", "util-linux"),
            
            # Compilers and Language Runtimes
            ("https://github.com/llvm/llvm-project.git", "llvm-project"),
            ("https://github.com/gcc-mirror/gcc.git", "gcc"),
            ("https://github.com/rust-lang/rust.git", "rust"),
            ("https://github.com/golang/go.git", "go"),
            ("https://github.com/python/cpython.git", "cpython"),
            ("https://github.com/nodejs/node.git", "nodejs"),
            ("https://github.com/openjdk/jdk.git", "openjdk"),
            ("https://github.com/microsoft/TypeScript.git", "TypeScript"), # ADDED
            ("https://github.com/JuliaLang/julia.git", "julia"),
            ("https://github.com/elixir-lang/elixir.git", "elixir"),
            ("https://github.com/erlang/otp.git", "erlang-otp"),
            ("https://github.com/scala/scala.git", "scala"),
            ("https://github.com/clojure/clojure.git", "clojure"),
            ("https://github.com/apple/swift.git", "swift"),
            ("https://github.com/JetBrains/kotlin.git", "kotlin"),
            
            # Databases
            ("https://github.com/postgres/postgres.git", "postgres"),
            ("https://github.com/mysql/mysql-server.git", "mysql"),
            ("https://github.com/redis/redis.git", "redis"),
            ("https://github.com/mongodb/mongo.git", "mongodb"),
            ("https://github.com/sqlite/sqlite.git", "sqlite"),
            ("https://github.com/opensearch-project/OpenSearch.git", "opensearch"), # REPLACED elasticsearch
            
            # Web Servers and Networking
            ("https://github.com/apache/httpd.git", "apache-httpd"),
            ("https://github.com/nginx/nginx.git", "nginx"),
            ("https://github.com/curl/curl.git", "curl"),
            ("https://github.com/openssl/openssl.git", "openssl"),
            
            # Web Frameworks & Libraries
            ("https://github.com/facebook/react.git", "react"),
            ("https://github.com/angular/angular.git", "angular"),
            ("https://github.com/vuejs/vue.git", "vue"),
            ("https://github.com/django/django.git", "django"),
            ("https://github.com/pallets/flask.git", "flask"),
            ("https://github.com/laravel/laravel.git", "laravel"),
            ("https://github.com/dotnet/aspnetcore.git", "aspnetcore"), # ADDED
            
            # Container and Cloud Technologies
            ("https://github.com/kubernetes/kubernetes.git", "kubernetes"),
            ("https://github.com/docker/cli.git", "docker-cli"),
            ("https://github.com/containerd/containerd.git", "containerd"),
            ("https://github.com/etcd-io/etcd.git", "etcd"),
            
            # Machine Learning and AI
            ("https://github.com/tensorflow/tensorflow.git", "tensorflow"),
            ("https://github.com/pytorch/pytorch.git", "pytorch"),
            ("https://github.com/scikit-learn/scikit-learn.git", "scikit-learn"),
            ("https://github.com/apache/spark.git", "apache-spark"),
            ("https://github.com/huggingface/transformers.git", "transformers"),
            
            # Graphics, Gaming, and Mobile
            ("https://github.com/godotengine/godot.git", "godot"),
            ("https://github.com/microsoft/DirectX-Graphics-Samples.git", "directx-samples"),
            ("https://github.com/Unity-Technologies/UnityCsReference.git", "unity-cs-reference"),
            ("https://github.com/facebook/react-native.git", "react-native"),
            ("https://github.com/flutter/flutter.git", "flutter"),

            # Major Applications and Codebases
            ("https://github.com/chromium/chromium.git", "chromium"),
            ("https://github.com/microsoft/vscode.git", "vscode"),
            ("https://github.com/electron/electron.git", "electron"),
            
            # Operating Systems
            ("https://github.com/freebsd/freebsd-src.git", "freebsd"),
            ("https://github.com/openbsd/src.git", "openbsd"),
            
            # Build Systems and Tools
            ("https://github.com/bazelbuild/bazel.git", "bazel"),
            ("https://github.com/Kitware/CMake.git", "cmake"),
            ("https://github.com/mesonbuild/meson.git", "meson"),
            
            # Infrastructure and DevOps
            ("https://github.com/apache/kafka.git", "apache-kafka"),
            ("https://github.com/apache/cassandra.git", "apache-cassandra"),
            ("https://github.com/apache/hadoop.git", "apache-hadoop"),
            ("https://github.com/hashicorp/terraform.git", "terraform"),
            ("https://github.com/hashicorp/vault.git", "vault"),
            ("https://github.com/prometheus/prometheus.git", "prometheus"),
            ("https://github.com/grafana/grafana.git", "grafana"),
            ("https://github.com/ansible/ansible.git", "ansible"),
        ])))

    def clone_repository(self, repo_url, repo_name):
        """Clone or update a single repository with improved error handling."""
        repo_path = self.base_dir / repo_name
        
        try:
            if repo_path.exists() and (repo_path / ".git").is_dir():
                logger.info(f"Repository '{repo_name}' already exists, updating...")
                # Fetch the latest changes from the remote
                subprocess.run(
                    ["git", "fetch", "--prune"],
                    cwd=repo_path, check=True, capture_output=True, text=True, timeout=3600
                )
                # Force the local branch to match the remote, discarding local changes/untracked files
                result = subprocess.run(
                    ["git", "reset", "--hard", "origin/HEAD"], # Adjust if the main branch isn't HEAD
                    cwd=repo_path, capture_output=True, text=True, timeout=3600
                )
                if result.returncode == 0:
                    logger.info(f"Successfully updated '{repo_name}'")
                    return repo_name, "updated", str(repo_path)
                else:
                    logger.warning(f"Failed to hard reset/update '{repo_name}': {result.stderr.strip()}")
                    return repo_name, "update_failed", result.stderr.strip()
            else:
                logger.info(f"Cloning '{repo_name}' from {repo_url}...")
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
                    capture_output=True, text=True, timeout=7200
                )
                if result.returncode == 0:
                    logger.info(f"Successfully cloned '{repo_name}'")
                    return repo_name, "cloned", str(repo_path)
                else:
                    logger.error(f"Failed to clone '{repo_name}': {result.stderr.strip()}")
                    return repo_name, "failed", result.stderr.strip()
                    
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while processing '{repo_name}'")
            return repo_name, "timeout", "Operation timed out"
        except subprocess.CalledProcessError as e:
            logger.error(f"A git command failed for '{repo_name}': {e.stderr}")
            return repo_name, "git_error", e.stderr
        except Exception as e:
            logger.error(f"Unexpected error with '{repo_name}': {str(e)}")
            return repo_name, "error", str(e)

    def index_repository_files(self, repo_path_str):
        """
        Recursively finds all files in a repository and writes their
        absolute paths to a central index file with a unique number.
        This method is thread-safe.
        """
        repo_path = Path(repo_path_str)
        logger.info(f"Starting to index files for '{repo_path.name}'...")
        
        try:
            files_to_index = [f for f in repo_path.rglob('*') if f.is_file()]
            
            if not files_to_index:
                logger.warning(f"No files found to index in '{repo_path.name}'.")
                return

            with self.index_lock:
                with open(self.index_file_path, 'a', encoding='utf-8') as f:
                    for file_path in files_to_index:
                        self.file_index_counter += 1
                        f.write(f"{self.file_index_counter} : {file_path.absolute()}\n")
            
            logger.info(f"Finished indexing {len(files_to_index)} files for '{repo_path.name}'.")

        except Exception as e:
            logger.error(f"Could not index files for '{repo_path.name}': {e}")

    def get_repo_stats(self, repo_path_str):
        """Get statistics about a cloned repository."""
        repo_path = Path(repo_path_str)
        try:
            code_extensions = {
                '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.py', '.js', '.ts', 
                '.java', '.go', '.rs', '.php', '.rb', '.cs', '.scala', '.kt', 
                '.swift', '.m', '.mm', '.sh', '.pl', '.lua', '.r', '.sql', '.html',
                '.css', '.xml', '.json', '.yml', '.yaml', '.md'
            }
            
            file_count = 0
            code_file_count = 0
            total_size = 0
            
            for file_path in repo_path.rglob('*'):
                if '.git' in file_path.parts:
                    continue
                if file_path.is_file():
                    file_count += 1
                    total_size += file_path.stat().st_size
                    if file_path.suffix.lower() in code_extensions:
                        code_file_count += 1
            
            return {
                'total_files': file_count,
                'code_files': code_file_count,
                'total_size_gb': total_size / (1024 ** 3)
            }
        except Exception as e:
            logger.warning(f"Could not get stats for {repo_path.name}: {e}")
            return {'total_files': 0, 'code_files': 0, 'total_size_gb': 0}
    
    def clone_all_repositories(self):
        """Clone all repositories using a thread pool for parallel execution."""
        repo_count = len(self.repositories)
        logger.info(f"Starting to process {repo_count} repositories using {self.max_workers} workers.")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="Cloner") as executor:
            future_to_repo = {
                executor.submit(self.clone_repository, repo_url, repo_name): (repo_url, repo_name)
                for repo_url, repo_name in self.repositories
            }
            
            for i, future in enumerate(as_completed(future_to_repo)):
                repo_url, repo_name = future_to_repo[future]
                logger.info(f"--- Progress: {i + 1}/{repo_count} futures completed ---")
                try:
                    name, status, detail = future.result()
                    
                    if status in ["cloned", "updated"]:
                        self.cloned_repos.append((name, status, detail))
                        
                        repo_path = detail
                        stats = self.get_repo_stats(repo_path)
                        logger.info(f"Stats for '{name}': {stats['total_files']:,} files, {stats['code_files']:,} code files, {stats['total_size_gb']:.3f} GB")
                        
                        self.index_repository_files(repo_path)
                    else:
                        self.failed_repos.append((name, status, detail))
                        
                except Exception as e:
                    logger.error(f"Error processing future for '{repo_name}': {e}")
                    self.failed_repos.append((repo_name, "future_error", str(e)))
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.print_summary(duration)
        
    def print_summary(self, duration):
        """Prints a detailed summary of the cloning and indexing process."""
        logger.info(f"\n{'='*80}")
        logger.info("CLONING AND INDEXING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total repositories to process: {len(self.repositories)}")
        logger.info(f"Successfully processed: {len(self.cloned_repos)}")
        logger.info(f"Failed to process: {len(self.failed_repos)}")
        logger.info(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        if self.failed_repos:
            logger.info("\n--- Failed Repositories ---")
            for repo_name, status, error in self.failed_repos:
                logger.warning(f"  - {repo_name}: Status='{status}', Reason='{error}'")
        
        total_stats = {'total_files': 0, 'code_files': 0, 'total_size_gb': 0}
        for repo_name, status, repo_path in self.cloned_repos:
            if status in ["cloned", "updated"]:
                stats = self.get_repo_stats(repo_path)
                for key in total_stats:
                    total_stats[key] += stats[key]
        
        logger.info(f"Total files indexed: {self.file_index_counter:,}")
        logger.info(f"Total code files found: {total_stats['code_files']:,}")
        logger.info(f"Total size on disk: {total_stats['total_size_gb']:.3f} GB")
        logger.info(f"{'='*80}")

def main():
    cloner = RepoCloner(base_dir="./repositories", max_workers=8)
    
    print("Starting repository cloning and indexing process...")
    print(f"Repositories will be cloned to: {cloner.base_dir.absolute()}")
    print(f"A full index of all files will be created at: {cloner.index_file_path.absolute()}")
    
    cloner.clone_all_repositories()
    
    print("\nProcess completed. Check 'repo_cloning.log' for detailed logs.")

if __name__ == "__main__":
    main()