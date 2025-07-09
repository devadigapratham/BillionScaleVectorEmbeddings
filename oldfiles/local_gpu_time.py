"""
script to test 10k embeddings locally and project to 1 billion
"""

import os
import time
import torch
from sentence_transformers import SentenceTransformer
import glob
import random

class EmbeddingTimer:
    def __init__(self):
        self.embedding_datasets_path = "embedding_datasets"
        self.models = {
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768
        }
        self.test_size = 10000
        self.target_size = 1_000_000_000
        
    def get_gpu_info(self):
        """Quick GPU check"""
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            print(f"GPU: {gpu.name} ({gpu.total_memory / (1024**3):.1f} GB)")
            return 'cuda'
        else:
            print("Using CPU (no CUDA available)")
            return 'cpu'
    
    def extract_code_snippets(self):
        """Extract code snippets from GCC codebase"""
        print(f"Extracting {self.test_size} code snippets from GCC...")
        
        gcc_path = os.path.join(self.embedding_datasets_path, "gcc")
        if not os.path.exists(gcc_path):
            raise FileNotFoundError(f"GCC directory not found at {gcc_path}")
        
        code_snippets = []
        # Focus on C/C++ files in GCC
        extensions = ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp']
        
        print(f"Scanning GCC directory: {gcc_path}")
        
        for ext in extensions:
            pattern = os.path.join(gcc_path, f"**/*{ext}")
            files = glob.glob(pattern, recursive=True)
            print(f"Found {len(files)} {ext} files")
            
            for file_path in files:
                if len(code_snippets) >= self.test_size:
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
                            if current_chunk:  # Add to existing chunk
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
                        
                        # Save chunk when we have a complete function or reach size limit
                        if (not in_function and len(current_chunk) >= 5) or len(current_chunk) >= 25:
                            chunk_text = '\n'.join(current_chunk)
                            if len(chunk_text.strip()) > 100:  # Minimum meaningful size
                                code_snippets.append(chunk_text)
                                if len(code_snippets) >= self.test_size:
                                    print(f"Extracted {len(code_snippets)} code snippets from GCC")
                                    return code_snippets[:self.test_size]
                            current_chunk = []
                            brace_count = 0
                            in_function = False
                    
                    # Don't forget the last chunk
                    if current_chunk and len('\n'.join(current_chunk).strip()) > 100:
                        code_snippets.append('\n'.join(current_chunk))
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
            
            if len(code_snippets) >= self.test_size:
                break
        
        if len(code_snippets) < self.test_size:
            print(f"Warning: Only found {len(code_snippets)} code snippets, less than requested {self.test_size}")
        
        print(f"Successfully extracted {len(code_snippets)} code snippets from GCC")
        return code_snippets[:self.test_size]
    
    def time_embeddings(self, model_name, code_snippets):
        """Time embedding generation for a model"""
        print(f"\n{'='*50}")
        print(f"TESTING: {model_name}")
        print(f"{'='*50}")
        
        # Load model
        print("Loading model...")
        start_load = time.time()
        model = SentenceTransformer(f'sentence-transformers/{model_name}')
        
        device = self.get_gpu_info()
        model = model.to(device)
        load_time = time.time() - start_load
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Find optimal batch size quickly
        batch_sizes = [32, 64, 128, 256] if device == 'cuda' else [8, 16, 32]
        best_batch = 32
        best_time = float('inf')
        
        print(f"Finding optimal batch size...")
        for batch_size in batch_sizes:
            try:
                # Test with smaller sample
                test_sample = code_snippets[:min(1000, len(code_snippets))]
                
                start = time.time()
                _ = model.encode(test_sample, batch_size=batch_size, show_progress_bar=False)
                if device == 'cuda':
                    torch.cuda.synchronize()
                test_time = time.time() - start
                
                throughput = len(test_sample) / test_time
                print(f"  Batch {batch_size}: {throughput:.1f} embeddings/sec")
                
                if test_time < best_time:
                    best_time = test_time
                    best_batch = batch_size
                    
            except Exception as e:
                print(f"  Batch {batch_size}: Failed ({str(e)[:50]})")
                break
        
        print(f"Using optimal batch size: {best_batch}")
        
        # Full timing test
        print(f"\nGenerating {len(code_snippets)} embeddings...")
        
        # Warmup
        warmup_sample = code_snippets[:min(100, len(code_snippets))]
        _ = model.encode(warmup_sample, batch_size=best_batch, show_progress_bar=False)
        
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Actual timing
        start_time = time.time()
        embeddings = model.encode(
            code_snippets, 
            batch_size=best_batch,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        throughput = len(code_snippets) / total_time
        
        return {
            'total_time': total_time,
            'throughput': throughput,
            'batch_size': best_batch,
            'embedding_dims': embeddings.shape[1],
            'device': device
        }
    
    def project_to_billion(self, results):
        """Project timing to 1 billion embeddings"""
        print(f"\n{'='*60}")
        print("BILLION EMBEDDING PROJECTIONS")
        print(f"{'='*60}")
        
        print(f"Based on {self.test_size:,} embeddings test")
        print(f"Projecting to {self.target_size:,} embeddings")
        print(f"Scale factor: {self.target_size // self.test_size:,}x")
        
        projections = {}
        
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"  Test throughput: {result['throughput']:.1f} embeddings/sec")
            print(f"  Optimal batch size: {result['batch_size']}")
            print(f"  Embedding dimensions: {result['embedding_dims']}")
            
            # Calculate projections
            total_seconds = self.target_size / result['throughput']
            total_minutes = total_seconds / 60
            total_hours = total_seconds / 3600
            total_days = total_hours / 24
            
            projections[model_name] = {
                'seconds': total_seconds,
                'hours': total_hours,
                'days': total_days,
                'throughput': result['throughput']
            }
            
            print(f"  Time for 1B embeddings:")
            print(f"    {total_seconds:,.0f} seconds")
            print(f"    {total_minutes:,.0f} minutes")  
            print(f"    {total_hours:,.1f} hours")
            print(f"    {total_days:.1f} days")
                    
        return projections
    
    def run_test(self):
        """Run the complete test"""
        print("Local GPU Embedding Timer")
        print("=" * 50)
        
        # Extract code from GCC
        code_snippets = self.extract_code_snippets()
        
        # Test both models
        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.time_embeddings(model_name, code_snippets)
        
        # Project to billion
        projections = self.project_to_billion(results)
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Throughput/sec':<15} {'Hours for 1B':<12} {'Days':<8}")
        print("-" * 60)
        
        for model_name in self.models.keys():
            throughput = results[model_name]['throughput']
            hours = projections[model_name]['hours']
            days = projections[model_name]['days']
            print(f"{model_name:<20} {throughput:<15.1f} {hours:<12.1f} {days:<8.1f}")
        
        # Best option
        fastest = min(projections.keys(), key=lambda x: projections[x]['hours'])
        print(f"\nFastest option: {fastest} ({projections[fastest]['hours']:.1f} hours)")

def main():
    timer = EmbeddingTimer()
    timer.run_test()

if __name__ == "__main__":
    main()