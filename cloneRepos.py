#!/usr/bin/env python3

import os
import subprocess
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('repo_cloning.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RepoCloner:
    def __init__(self, base_dir="./repositories", max_workers=8):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.cloned_repos = []
        self.failed_repos = []
        
        # Major repositories to clone
        self.repositories = [
            # Linux Kernel and Core Systems
            ("https://github.com/torvalds/linux.git", "linux"),
            ("https://github.com/systemd/systemd.git", "systemd"),
            ("https://github.com/util-linux/util-linux.git", "util-linux"),
            ("https://github.com/systemd/systemd.git", "systemd"),
            
            # Compilers and Language Tools
            ("https://github.com/llvm/llvm-project.git", "llvm-project"),
            ("https://github.com/gcc-mirror/gcc.git", "gcc"),
            ("https://github.com/rust-lang/rust.git", "rust"),
            ("https://github.com/golang/go.git", "go"),
            ("https://github.com/python/cpython.git", "cpython"),
            ("https://github.com/nodejs/node.git", "nodejs"),
            ("https://github.com/openjdk/jdk.git", "openjdk"),
            
            # Databases
            ("https://github.com/postgres/postgres.git", "postgres"),
            ("https://github.com/mysql/mysql-server.git", "mysql"),
            ("https://github.com/redis/redis.git", "redis"),
            ("https://github.com/mongodb/mongo.git", "mongodb"),
            ("https://github.com/sqlite/sqlite.git", "sqlite"),
            ("https://github.com/elastic/elasticsearch.git", "elasticsearch"),
            
            # Web Servers and Networking
            ("https://github.com/apache/httpd.git", "apache-httpd"),
            ("https://github.com/nginx/nginx.git", "nginx"),
            ("https://github.com/curl/curl.git", "curl"),
            ("https://github.com/openssl/openssl.git", "openssl"),
            
            # Container and Cloud Technologies
            ("https://github.com/kubernetes/kubernetes.git", "kubernetes"),
            ("https://github.com/docker/docker-ce.git", "docker"),
            ("https://github.com/containerd/containerd.git", "containerd"),
            ("https://github.com/etcd-io/etcd.git", "etcd"),
            
            # Machine Learning and AI
            ("https://github.com/tensorflow/tensorflow.git", "tensorflow"),
            ("https://github.com/pytorch/pytorch.git", "pytorch"),
            ("https://github.com/scikit-learn/scikit-learn.git", "scikit-learn"),
            ("https://github.com/apache/spark.git", "apache-spark"),
            ("https://github.com/huggingface/transformers.git", "transformers"),
            
            # Graphics and Gaming
            ("https://github.com/godotengine/godot.git", "godot"),
            ("https://github.com/blender/blender.git", "blender"),
            ("https://github.com/microsoft/DirectX-Graphics-Samples.git", "directx-samples"),
            
            # Additional Large Codebases
            ("https://github.com/chromium/chromium.git", "chromium"),
            ("https://github.com/microsoft/vscode.git", "vscode"),
            ("https://github.com/facebook/react.git", "react"),
            ("https://github.com/angular/angular.git", "angular"),
            ("https://github.com/vuejs/vue.git", "vue"),
            ("https://github.com/facebook/react-native.git", "react-native"),
            ("https://github.com/electron/electron.git", "electron"),
            
            # Operating Systems
            ("https://github.com/freebsd/freebsd-src.git", "freebsd"),
            ("https://github.com/openbsd/src.git", "openbsd"),
            
            # Build Systems and Tools
            ("https://github.com/bazelbuild/bazel.git", "bazel"),
            ("https://github.com/Kitware/CMake.git", "cmake"),
            ("https://github.com/mesonbuild/meson.git", "meson"),
            
            # More Programming Languages
            ("https://github.com/dotnet/runtime.git", "dotnet-runtime"),
            ("https://github.com/JuliaLang/julia.git", "julia"),
            ("https://github.com/elixir-lang/elixir.git", "elixir"),
            ("https://github.com/erlang/otp.git", "erlang-otp"),
            ("https://github.com/scala/scala.git", "scala"),
            ("https://github.com/clojure/clojure.git", "clojure"),
            
            # Game Engines and Graphics
            ("https://github.com/UnrealEngine/UnrealEngine.git", "unreal-engine"),  # Requires Epic Games account
            ("https://github.com/unity3d-jp/UnityCsReference.git", "unity-cs-reference"),
            
            # Additional Major Projects
            ("https://github.com/apache/kafka.git", "apache-kafka"),
            ("https://github.com/apache/cassandra.git", "apache-cassandra"),
            ("https://github.com/apache/hadoop.git", "apache-hadoop"),
            ("https://github.com/hashicorp/terraform.git", "terraform"),
            ("https://github.com/hashicorp/vault.git", "vault"),
            ("https://github.com/prometheus/prometheus.git", "prometheus"),
            ("https://github.com/grafana/grafana.git", "grafana"),
        ]
    
    def clone_repository(self, repo_url, repo_name):
        """Clone a single repository with error handling and progress tracking."""
        repo_path = self.base_dir / repo_name
        
        try:
            if repo_path.exists():
                logger.info(f"Repository {repo_name} already exists, updating...")
                result = subprocess.run(
                    ["git", "pull", "--ff-only"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                if result.returncode == 0:
                    logger.info(f"Successfully updated {repo_name}")
                    return repo_name, "updated", str(repo_path)
                else:
                    logger.warning(f"Failed to update {repo_name}: {result.stderr}")
                    return repo_name, "update_failed", str(repo_path)
            else:
                logger.info(f"Cloning {repo_name} from {repo_url}...")
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
                    capture_output=True,
                    text=True,
                    timeout=7200  # 2 hour timeout for large repos
                )
                
                if result.returncode == 0:
                    logger.info(f"Successfully cloned {repo_name}")
                    return repo_name, "cloned", str(repo_path)
                else:
                    logger.error(f"Failed to clone {repo_name}: {result.stderr}")
                    return repo_name, "failed", result.stderr
                    
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while cloning {repo_name}")
            return repo_name, "timeout", "Operation timed out"
        except Exception as e:
            logger.error(f"Unexpected error cloning {repo_name}: {str(e)}")
            return repo_name, "error", str(e)
    
    def get_repo_stats(self, repo_path):
        """Get statistics about a cloned repository."""
        try:
            # Count files by extension
            code_extensions = {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.py', '.js', '.ts', 
                             '.java', '.go', '.rs', '.php', '.rb', '.cs', '.scala', '.kt', 
                             '.swift', '.m', '.mm', '.sh', '.pl', '.lua', '.r', '.sql'}
            
            file_count = 0
            code_file_count = 0
            total_size = 0
            
            for file_path in Path(repo_path).rglob('*'):
                if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                    file_count += 1
                    total_size += file_path.stat().st_size
                    if file_path.suffix.lower() in code_extensions:
                        code_file_count += 1
            
            return {
                'total_files': file_count,
                'code_files': code_file_count,
                'total_size_mb': total_size / (1024 * 1024)
            }
        except Exception as e:
            logger.warning(f"Could not get stats for {repo_path}: {e}")
            return {'total_files': 0, 'code_files': 0, 'total_size_mb': 0}
    
    def clone_all_repositories(self):
        """Clone all repositories using thread pool for parallel execution."""
        logger.info(f"Starting to clone {len(self.repositories)} repositories using {self.max_workers} workers")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all cloning tasks
            future_to_repo = {
                executor.submit(self.clone_repository, repo_url, repo_name): (repo_url, repo_name)
                for repo_url, repo_name in self.repositories
            }
            
            # Process completed tasks
            for future in as_completed(future_to_repo):
                repo_url, repo_name = future_to_repo[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result[1] in ["cloned", "updated"]:
                        self.cloned_repos.append(result)
                        # Get repository statistics
                        stats = self.get_repo_stats(result[2])
                        logger.info(f"Repository {result[0]} stats: {stats}")
                    else:
                        self.failed_repos.append(result)
                        
                except Exception as e:
                    logger.error(f"Error processing {repo_name}: {e}")
                    self.failed_repos.append((repo_name, "processing_error", str(e)))
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"CLONING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total repositories: {len(self.repositories)}")
        logger.info(f"Successfully processed: {len(self.cloned_repos)}")
        logger.info(f"Failed: {len(self.failed_repos)}")
        logger.info(f"Total time: {duration:.2f} seconds")
        
        if self.failed_repos:
            logger.info(f"\nFailed repositories:")
            for repo_name, status, error in self.failed_repos:
                logger.info(f"  - {repo_name}: {status} - {error}")
        
        # Calculate total statistics
        total_stats = {'total_files': 0, 'code_files': 0, 'total_size_mb': 0}
        for repo_name, status, repo_path in self.cloned_repos:
            if status in ["cloned", "updated"]:
                stats = self.get_repo_stats(repo_path)
                for key in total_stats:
                    total_stats[key] += stats[key]
        
        logger.info(f"\nOverall Statistics:")
        logger.info(f"Total files: {total_stats['total_files']:,}")
        logger.info(f"Code files: {total_stats['code_files']:,}")
        logger.info(f"Total size: {total_stats['total_size_mb']:.2f} MB")
        
        return results

def main():
    """Main function to run the repository cloning process."""
    cloner = RepoCloner(base_dir="./repositories", max_workers=8)
    
    print("Starting repository cloning process...")
    print(f"Repositories will be cloned to: {cloner.base_dir.absolute()}")
    
    # Create the base directory if it doesn't exist
    cloner.base_dir.mkdir(exist_ok=True)
    
    # Start cloning
    results = cloner.clone_all_repositories()
    
    print(f"\nCloning process completed. Check 'repo_cloning.log' for detailed logs.")
    print(f"Successfully processed repositories are stored in: {cloner.base_dir.absolute()}")
    
    return results

if __name__ == "__main__":
    main()