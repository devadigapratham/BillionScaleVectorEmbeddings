"""
Modal Cloud GPU script to test 10k embeddings on A100 40GB and project to 1 billion
Version without synthetic data - requires real GCC codebase
"""

import modal
import time
import os
import glob
import tempfile
import shutil
from typing import List, Dict, Any

# Modal setup
app = modal.App("embedding-timer")

# Define the Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch",
        "sentence-transformers",
        "transformers",
        "numpy",
        "tqdm"
    ])
    # Use the new add_local_dir method instead of Mount (fixes deprecation warning)
    .add_local_dir("embedding_datasets/gcc_subset", remote_path="/gcc")
)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
    memory=32768,
)
def extract_gcc_code(test_size: int = 10000) -> List[str]:
    """Extract code snippets from uploaded GCC codebase"""
    print(f"Extracting {test_size} code snippets from GCC...")
    
    code_snippets = []
    gcc_path = "/gcc"
    
    if not os.path.exists(gcc_path):
        raise FileNotFoundError(f"GCC directory not found at {gcc_path}. Please ensure 'embedding_datasets/gcc_subset' exists locally.")
    
    extensions = ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp']
    
    print(f"Scanning GCC directory: {gcc_path}")
    
    for ext in extensions:
        pattern = os.path.join(gcc_path, f"**/*{ext}")
        files = glob.glob(pattern, recursive=True)
        print(f"Found {len(files)} {ext} files")
        
        for file_path in files:
            if len(code_snippets) >= test_size:
                break
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Skip very small files
                if len(content.strip()) < 200:
                    continue
                
                # Extract meaningful code chunks
                lines = content.split('\n')
                current_chunk = []
                in_function = False
                brace_count = 0
                
                for line in lines:
                    stripped_line = line.strip()
                    
                    # Skip empty lines and comments-only lines
                    if not stripped_line or stripped_line.startswith('//') or stripped_line.startswith('/*'):
                        if current_chunk:
                            current_chunk.append(line)
                        continue
                    
                    current_chunk.append(line)
                    
                    # Track function boundaries
                    if '{' in line:
                        brace_count += line.count('{')
                        in_function = True
                    if '}' in line:
                        brace_count -= line.count('}')
                        if brace_count <= 0:
                            in_function = False
                    
                    # Save chunk when complete
                    if (not in_function and len(current_chunk) >= 5) or len(current_chunk) >= 25:
                        chunk_text = '\n'.join(current_chunk)
                        if len(chunk_text.strip()) > 100:
                            code_snippets.append(chunk_text)
                            if len(code_snippets) >= test_size:
                                print(f"Extracted {len(code_snippets)} code snippets from GCC")
                                return code_snippets[:test_size]
                        current_chunk = []
                        brace_count = 0
                        in_function = False
                
                # Last chunk
                if current_chunk and len('\n'.join(current_chunk).strip()) > 100:
                    code_snippets.append('\n'.join(current_chunk))
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        if len(code_snippets) >= test_size:
            break
    
    if len(code_snippets) < test_size:
        print(f"Warning: Only extracted {len(code_snippets)} code snippets, requested {test_size}")
    
    print(f"Successfully extracted {len(code_snippets)} code snippets from GCC")
    return code_snippets[:test_size]

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
    memory=32768,
)
def benchmark_model_on_a100(model_name: str, code_snippets: List[str]) -> Dict[str, Any]:
    """Benchmark embedding generation on A100 40GB"""
    import torch
    from sentence_transformers import SentenceTransformer
    
    print(f"\n{'='*50}")
    print(f"BENCHMARKING ON A100 40GB: {model_name}")
    print(f"{'='*50}")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Load model
    print("Loading model...")
    start_load = time.time()
    model = SentenceTransformer(f'sentence-transformers/{model_name}')
    model = model.to('cuda')
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Test batch sizes optimized for A100 40GB
    batch_sizes = [64, 128, 256, 512, 1024, 2048]
    best_batch = 128
    best_throughput = 0
    
    print(f"Optimizing batch size for A100...")
    
    batch_results = {}
    
    for batch_size in batch_sizes:
        try:
            print(f"Testing batch size: {batch_size}")
            
            # Test with 2k samples to find optimal batch size
            test_sample = code_snippets[:min(2000, len(code_snippets))]
            
            # Warmup
            warmup_samples = test_sample[:min(batch_size * 2, len(test_sample))]
            _ = model.encode(warmup_samples, batch_size=batch_size, show_progress_bar=False)
            torch.cuda.synchronize()
            
            # Clear memory stats
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
            
            # Timing test
            start_time = time.time()
            embeddings = model.encode(
                test_sample,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            torch.cuda.synchronize()
            end_time = time.time()
            
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            test_time = end_time - start_time
            throughput = len(test_sample) / test_time
            memory_used = (end_memory - start_memory) / (1024**2)  # MB
            peak_memory_gb = peak_memory / (1024**3)  # GB
            
            batch_results[batch_size] = {
                'time': test_time,
                'throughput': throughput,
                'memory_mb': memory_used,
                'peak_memory_gb': peak_memory_gb
            }
            
            print(f"  Throughput: {throughput:.1f} embeddings/sec")
            print(f"  Peak memory: {peak_memory_gb:.1f} GB")
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch = batch_size
                
        except Exception as e:
            print(f"  Error with batch size {batch_size}: {e}")
            # If we hit memory issues, stop testing larger batches
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                print("  Memory limit reached for A100 40GB")
                break
    
    print(f"Optimal batch size: {best_batch} (throughput: {best_throughput:.1f}/sec)")
    
    # Full benchmark with all samples
    print(f"\nFull benchmark with {len(code_snippets)} embeddings...")
    
    # Warmup
    warmup_sample = code_snippets[:min(500, len(code_snippets))]
    _ = model.encode(warmup_sample, batch_size=best_batch, show_progress_bar=False)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Full timing
    start_time = time.time()
    embeddings = model.encode(
        code_snippets,
        batch_size=best_batch,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    throughput = len(code_snippets) / total_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    
    print(f"Final Results:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Throughput: {throughput:.1f} embeddings/sec")
    print(f"  Peak memory: {peak_memory_gb:.1f} GB")
    print(f"  Embedding shape: {embeddings.shape}")
    
    return {
        'total_time': total_time,
        'throughput': throughput,
        'batch_size': best_batch,
        'embedding_dims': embeddings.shape[1],
        'peak_memory_gb': peak_memory_gb,
        'batch_results': batch_results
    }

@app.function(
    image=image,
    timeout=1800,
    memory=8192,
)
def project_to_billion(results: Dict[str, Dict[str, Any]], test_size: int, target_size: int):
    """Project timing to 1 billion embeddings"""
    print(f"\n{'='*60}")
    print("BILLION EMBEDDING PROJECTIONS (A100 40GB)")
    print(f"{'='*60}")
    
    print(f"Based on {test_size:,} embeddings test")
    print(f"Projecting to {target_size:,} embeddings")
    print(f"Scale factor: {target_size // test_size:,}x")
    
    projections = {}
    
    for model_name, result in results.items():
        print(f"\n{model_name} on A100 40GB:")
        print(f"  Test throughput: {result['throughput']:.1f} embeddings/sec")
        print(f"  Optimal batch size: {result['batch_size']}")
        print(f"  Peak memory usage: {result['peak_memory_gb']:.1f} GB")
        print(f"  Embedding dimensions: {result['embedding_dims']}")
        
        # Calculate projections
        total_seconds = target_size / result['throughput']
        total_minutes = total_seconds / 60
        total_hours = total_seconds / 3600
        total_days = total_hours / 24
        
        # Cost estimation for Modal A100
        # Modal A100 40GB costs approximately $2.60/hour
        modal_cost_per_hour = 2.60
        total_cost = total_hours * modal_cost_per_hour
        
        projections[model_name] = {
            'seconds': total_seconds,
            'hours': total_hours,
            'days': total_days,
            'throughput': result['throughput'],
            'cost_usd': total_cost
        }
        
        print(f"  Time for 1B embeddings:")
        print(f"    {total_seconds:,.0f} seconds")
        print(f"    {total_minutes:,.0f} minutes")  
        print(f"    {total_hours:,.1f} hours")
        print(f"    {total_days:.1f} days")
        print(f"  💰 Estimated Modal cost: ${total_cost:,.2f}")
        
    return projections

@app.local_entrypoint()
def main():
    """Main function to run the complete benchmark"""
    print("Modal A100 40GB Embedding Timer")
    print("=" * 50)
    
    TEST_SIZE = 10000
    TARGET_SIZE = 1_000_000_000
    
    models = {
        'all-MiniLM-L6-v2': 384,
        'all-mpnet-base-v2': 768
    }
    
    print(f"🔧 REQUIREMENTS:")
    print(f"   1. Create directory: 'embedding_datasets/gcc_subset'")
    print(f"   2. Copy your GCC .c/.cpp/.h files there (under 125k files total)")
    print(f"   3. Run: modal run modal_gpu_time.py")
    print(f"   4. Script will fail if GCC directory is missing\n")
    
    # Extract code (requires real GCC directory)
    print("Extracting code samples from GCC...")
    code_snippets = extract_gcc_code.remote(TEST_SIZE)
    
    # Benchmark both models on A100
    results = {}
    for model_name in models.keys():
        print(f"\n🚀 Benchmarking {model_name} on A100 40GB...")
        results[model_name] = benchmark_model_on_a100.remote(model_name, code_snippets)
    
    # Project to billion
    projections = project_to_billion.remote(results, TEST_SIZE, TARGET_SIZE)
    
    # Summary
    print(f"\n{'='*70}")
    print("MODAL A100 40GB SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Throughput/sec':<15} {'Hours for 1B':<12} {'Cost (USD)':<12}")
    print("-" * 70)
    
    for model_name in models.keys():
        throughput = results[model_name]['throughput']
        hours = projections[model_name]['hours']
        cost = projections[model_name]['cost_usd']
        print(f"{model_name:<20} {throughput:<15.1f} {hours:<12.1f} ${cost:<11.2f}")
    
    # Best option
    fastest = min(projections.keys(), key=lambda x: projections[x]['hours'])
    cheapest = min(projections.keys(), key=lambda x: projections[x]['cost_usd'])
    
    print(f"\n🏆 Fastest: {fastest} ({projections[fastest]['hours']:.1f} hours)")
    print(f"💰 Cheapest: {cheapest} (${projections[cheapest]['cost_usd']:.2f})")

if __name__ == "__main__":
    main()