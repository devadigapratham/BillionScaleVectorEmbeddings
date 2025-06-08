"""
Generates 100k embeddings from codebases and test compression ratios
Compares different compression algorithms and measures size reduction
"""

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import glob
import random
import pickle
import gzip
import bz2
import lzma

class EmbeddingCompressionAnalyzer:
    def __init__(self):
        self.embedding_datasets_path = "embedding_datasets"
        self.models = {
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768
        }
        self.compression_algorithms = [
            'snappy',
            'gzip', 
            'brotli',
            'lz4',
            'zstd'
        ]
    
    def extract_code_snippets(self, max_snippets=100000):
        """Extract code snippets from all repositories"""
        print("Extracting code snippets from repositories...")
        
        code_snippets = []
        extensions = ['.py', '.cpp', '.c', '.h', '.hpp', '.java', '.js', '.ts', '.go', '.rs']
        
        for repo_dir in os.listdir(self.embedding_datasets_path):
            repo_path = os.path.join(self.embedding_datasets_path, repo_dir)
            if not os.path.isdir(repo_path):
                continue
                
            print(f"Processing repository: {repo_dir}")
            
            # Find all code files
            for ext in extensions:
                pattern = os.path.join(repo_path, f"**/*{ext}")
                files = glob.glob(pattern, recursive=True)
                
                for file_path in files[:100]:  # Limit files per extension per repo
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        # Split into functions/classes (simple heuristic)
                        lines = content.split('\n')
                        current_snippet = []
                        
                        for line in lines:
                            if any(keyword in line for keyword in ['def ', 'class ', 'function ', 'void ', 'int ', 'public ']):
                                if current_snippet and len(' '.join(current_snippet)) > 50:
                                    code_snippets.append(' '.join(current_snippet))
                                    if len(code_snippets) >= max_snippets:
                                        return code_snippets
                                current_snippet = [line]
                            else:
                                current_snippet.append(line)
                                if len(current_snippet) > 20:  # Limit snippet length
                                    if len(' '.join(current_snippet)) > 50:
                                        code_snippets.append(' '.join(current_snippet))
                                        if len(code_snippets) >= max_snippets:
                                            return code_snippets
                                    current_snippet = []
                                    
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
        
        # If we don't have enough, duplicate and modify existing snippets
        while len(code_snippets) < max_snippets:
            original = random.choice(code_snippets[:len(code_snippets)//2])
            # Add some variation
            modified = original + f" // variation_{len(code_snippets)}"
            code_snippets.append(modified)
        
        return code_snippets[:max_snippets]
    
    def generate_embeddings(self, code_snippets, model_name):
        """Generate embeddings using sentence transformers"""
        print(f"Generating embeddings with {model_name}...")
        
        model = SentenceTransformer(f'sentence-transformers/{model_name}')
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        embeddings = []
        
        for i in range(0, len(code_snippets), batch_size):
            batch = code_snippets[i:i+batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=True)
            embeddings.extend(batch_embeddings)
            print(f"Processed {min(i+batch_size, len(code_snippets))}/{len(code_snippets)} snippets")
        
        return np.array(embeddings)
    
    def save_baseline_formats(self, embeddings, model_name):
        """Save embeddings in different baseline formats for comparison"""
        formats_info = {}
        
        # NumPy format
        np_path = f"embeddings_{model_name}_numpy.npy"
        np.save(np_path, embeddings)
        formats_info['numpy'] = os.path.getsize(np_path)
        
        # Pickle format
        pickle_path = f"embeddings_{model_name}_pickle.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(embeddings, f)
        formats_info['pickle'] = os.path.getsize(pickle_path)
        
        # Compressed pickle formats
        for compressor, ext in [('gzip', 'gz'), ('bz2', 'bz2'), ('lzma', 'xz')]:
            comp_path = f"embeddings_{model_name}_pickle.{ext}"
            if compressor == 'gzip':
                with gzip.open(comp_path, 'wb') as f:
                    pickle.dump(embeddings, f)
            elif compressor == 'bz2':
                with bz2.open(comp_path, 'wb') as f:
                    pickle.dump(embeddings, f)
            elif compressor == 'lzma':
                with lzma.open(comp_path, 'wb') as f:
                    pickle.dump(embeddings, f)
            formats_info[f'pickle_{compressor}'] = os.path.getsize(comp_path)
        
        return formats_info
    
    def test_parquet_compression(self, embeddings, model_name):
        """Test different parquet compression algorithms"""
        print(f"Testing Parquet compression for {model_name}...")
        
        # Create DataFrame
        df_data = {}
        for i in range(embeddings.shape[1]):
            df_data[f'dim_{i}'] = embeddings[:, i]
        
        df = pd.DataFrame(df_data)
        
        compression_results = {}
        
        for compression in self.compression_algorithms:
            try:
                filename = f"embeddings_{model_name}_{compression}.parquet"
                
                # Write with compression
                table = pa.Table.from_pandas(df)
                pq.write_table(table, filename, compression=compression, compression_level=9)
                
                file_size = os.path.getsize(filename)
                
                compression_results[compression] = {
                    'file_size': file_size
                }
                
            except Exception as e:
                print(f"Warning: {compression} compression failed - {e}")
                compression_results[compression] = None
        
        return compression_results
    
    def analyze_and_report(self, baseline_sizes, parquet_results, model_name, embedding_dims):
        """Generate comprehensive analysis report"""
        print(f"\n{'='*50}")
        print(f"COMPRESSION ANALYSIS - {model_name}")
        print(f"Dataset: 100,000 vectors × {embedding_dims} dimensions")
        print(f"{'='*50}")
        
        # Get uncompressed size for reference
        uncompressed_mb = baseline_sizes['numpy'] / (1024 * 1024)
        print(f"\nUncompressed size: {uncompressed_mb:.1f} MB")
        
        # Parquet compression results
        print(f"\nCompression Results:")
        print(f"{'Algorithm':<12} {'Size (MB)':<12} {'Compression':<12} {'Space Saved'}")
        print("-" * 52)
        
        best_compression = None
        best_ratio = 0
        best_size = float('inf')
        
        for algo, results in parquet_results.items():
            if results is None:
                print(f"{algo:<12} {'Failed':<12} {'-':<12} {'-'}")
                continue
                
            size_mb = results['file_size'] / (1024 * 1024)
            ratio = baseline_sizes['numpy'] / results['file_size']
            space_saved = (1 - results['file_size'] / baseline_sizes['numpy']) * 100
            
            if results['file_size'] < best_size:
                best_size = results['file_size']
                best_ratio = ratio
                best_compression = algo
            
            print(f"{algo:<12} {size_mb:<12.1f} {ratio:<12.1f}x {space_saved:<11.1f}%")
        
        # Scale estimates
        if best_compression:
            print(f"\nBest: {best_compression} ({best_ratio:.1f}x compression)")
            
            # Estimates for larger datasets
            million_size_mb = (best_size * 10) / (1024 * 1024)
            billion_size_gb = (best_size * 10000) / (1024**3)
            
            print(f"\nStorage Estimates:")
            print(f"  1 million embeddings:  {million_size_mb:.1f} MB")
            print(f"  1 billion embeddings:  {billion_size_gb:.1f} GB")
        
        return {
            'best_algorithm': best_compression,
            'best_ratio': best_ratio,
            'best_size_mb': best_size / (1024 * 1024) if best_compression else None,
            'baseline_mb': uncompressed_mb
        }

def main():
    analyzer = EmbeddingCompressionAnalyzer()
    
    # Extract code snippets
    code_snippets = analyzer.extract_code_snippets(100000)
    print(f"Extracted {len(code_snippets)} code snippets")
    
    results = {}
    
    # Test both models
    for model_name, dims in analyzer.models.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING: {model_name}")
        print(f"{'='*60}")
        
        # Generate embeddings
        embeddings = analyzer.generate_embeddings(code_snippets, model_name)
        
        # Save baseline formats
        baseline_sizes = analyzer.save_baseline_formats(embeddings, model_name)
        
        # Test parquet compression
        parquet_results = analyzer.test_parquet_compression(embeddings, model_name)
        
        # Analyze and report
        analysis = analyzer.analyze_and_report(baseline_sizes, parquet_results, model_name, dims)
        results[model_name] = analysis
    
    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for model_name, analysis in results.items():
        if analysis['best_algorithm']:
            print(f"\n{model_name}:")
            print(f"  Best compression: {analysis['best_algorithm']}")
            print(f"  Ratio: {analysis['best_ratio']:.1f}x")
            print(f"  Size: {analysis['best_size_mb']:.1f} MB (from {analysis['baseline_mb']:.1f} MB)")

if __name__ == "__main__":
    main()