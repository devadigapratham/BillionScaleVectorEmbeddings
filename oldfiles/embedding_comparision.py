"""
Generates 100k embeddings from codebases and test compression ratios
Compares different compression algorithms and measures size reduction
WITH DATA INTEGRITY VERIFICATION - ensures no data is lost during compression
"""

import os
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import glob
import random
from pathlib import Path
import pickle
import gzip
import bz2
import lzma
import hashlib
from typing import Dict, Tuple, Any

class EmbeddingCompressionAnalyzer:
    def __init__(self):
        self.embedding_datasets_path = "embedding_datasets"
        self.models = {
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768
        }
        self.compression_algorithms = [
            'gzip', 
            'brotli',
            'lz4',
            'zstd'
        ]
        # Store original embeddings hash for verification
        self.original_embeddings_hash = {}
        self.verification_results = {}
    
    def calculate_embedding_hash(self, embeddings: np.ndarray) -> str:
        """Calculate SHA256 hash of embeddings for integrity verification"""
        return hashlib.sha256(embeddings.tobytes()).hexdigest()
    
    def verify_embeddings_integrity(self, original: np.ndarray, loaded: np.ndarray, 
                                  format_name: str, tolerance: float = 1e-10) -> Dict[str, Any]:
        """
        Comprehensive verification of embedding integrity
        
        Args:
            original: Original embeddings array
            loaded: Loaded embeddings array  
            format_name: Name of the format being tested
            tolerance: Floating point comparison tolerance
            
        Returns:
            Dictionary with verification results
        """
        verification = {
            'format': format_name,
            'shape_match': False,
            'dtype_match': False,
            'hash_match': False,
            'values_match': False,
            'max_difference': float('inf'),
            'mean_difference': float('inf'),
            'perfect_match': False,
            'error': None
        }
        
        try:
            # Check shapes
            verification['shape_match'] = original.shape == loaded.shape
            if not verification['shape_match']:
                verification['error'] = f"Shape mismatch: {original.shape} vs {loaded.shape}"
                return verification
            
            # Check data types
            verification['dtype_match'] = original.dtype == loaded.dtype
            
            # Calculate hash of loaded data
            loaded_hash = self.calculate_embedding_hash(loaded)
            original_hash = self.original_embeddings_hash.get(format_name.split('_')[0], '')
            verification['hash_match'] = loaded_hash == original_hash
            
            # Numerical comparison (handles floating point precision issues)
            if np.allclose(original, loaded, rtol=tolerance, atol=tolerance):
                verification['values_match'] = True
                verification['max_difference'] = np.max(np.abs(original - loaded))
                verification['mean_difference'] = np.mean(np.abs(original - loaded))
            else:
                verification['values_match'] = False
                verification['max_difference'] = np.max(np.abs(original - loaded))
                verification['mean_difference'] = np.mean(np.abs(original - loaded))
            
            # Perfect match check (exact equality)
            verification['perfect_match'] = np.array_equal(original, loaded)
            
        except Exception as e:
            verification['error'] = str(e)
        
        return verification
    
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
        
        embeddings_array = np.array(embeddings)
        
        # Store hash of original embeddings for verification
        self.original_embeddings_hash[model_name] = self.calculate_embedding_hash(embeddings_array)
        print(f"Original embeddings hash: {self.original_embeddings_hash[model_name][:16]}...")
        
        return embeddings_array
    
    def save_baseline_formats(self, embeddings, model_name):
        """Save embeddings in different baseline formats for comparison WITH VERIFICATION"""
        print(f"Saving baseline formats for {model_name} with verification...")
        formats_info = {}
        
        # NumPy format
        np_path = f"embeddings_{model_name}_numpy.npy"
        np.save(np_path, embeddings)
        formats_info['numpy'] = os.path.getsize(np_path)
        
        # Verify NumPy format
        loaded_numpy = np.load(np_path)
        verification = self.verify_embeddings_integrity(embeddings, loaded_numpy, f"{model_name}_numpy")
        self.verification_results[f"{model_name}_numpy"] = verification
        
        # Pickle format
        pickle_path = f"embeddings_{model_name}_pickle.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(embeddings, f)
        formats_info['pickle'] = os.path.getsize(pickle_path)
        
        # Verify Pickle format
        with open(pickle_path, 'rb') as f:
            loaded_pickle = pickle.load(f)
        verification = self.verify_embeddings_integrity(embeddings, loaded_pickle, f"{model_name}_pickle")
        self.verification_results[f"{model_name}_pickle"] = verification
        
        # Compressed pickle formats
        for compressor, ext in [('gzip', 'gz'), ('bz2', 'bz2'), ('lzma', 'xz')]:
            comp_path = f"embeddings_{model_name}_pickle.{ext}"
            
            # Save compressed
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
            
            # Verify compressed format
            try:
                if compressor == 'gzip':
                    with gzip.open(comp_path, 'rb') as f:
                        loaded_compressed = pickle.load(f)
                elif compressor == 'bz2':
                    with bz2.open(comp_path, 'rb') as f:
                        loaded_compressed = pickle.load(f)
                elif compressor == 'lzma':
                    with lzma.open(comp_path, 'rb') as f:
                        loaded_compressed = pickle.load(f)
                
                verification = self.verify_embeddings_integrity(
                    embeddings, loaded_compressed, f"{model_name}_pickle_{compressor}"
                )
                self.verification_results[f"{model_name}_pickle_{compressor}"] = verification
                
            except Exception as e:
                verification = {'format': f"{model_name}_pickle_{compressor}", 'error': str(e)}
                self.verification_results[f"{model_name}_pickle_{compressor}"] = verification
        
        return formats_info
    
    def test_parquet_compression(self, embeddings, model_name):
        """Test different parquet compression algorithms WITH VERIFICATION"""
        print(f"Testing Parquet compression for {model_name} with verification...")
        
        # Create DataFrame
        df_data = {}
        for i in range(embeddings.shape[1]):
            df_data[f'dim_{i}'] = embeddings[:, i]
        
        df = pd.DataFrame(df_data)
        
        compression_results = {}
        
        for compression in self.compression_algorithms:
            try:
                start_time = time.time()
                filename = f"embeddings_{model_name}_{compression}.parquet"
                
                # Write with compression
                table = pa.Table.from_pandas(df)
                pq.write_table(table, filename, compression=compression, compression_level=9)
                
                file_size = os.path.getsize(filename)
                write_time = time.time() - start_time
                
                # Test read time and verify data integrity
                start_time = time.time()
                loaded_table = pq.read_table(filename)
                loaded_df = loaded_table.to_pandas()
                read_time = time.time() - start_time
                
                # Convert back to numpy array for verification
                loaded_embeddings = loaded_df.values.astype(embeddings.dtype)
                
                # Verify data integrity
                verification = self.verify_embeddings_integrity(
                    embeddings, loaded_embeddings, f"{model_name}_{compression}"
                )
                self.verification_results[f"{model_name}_{compression}"] = verification
                
                compression_results[compression] = {
                    'file_size': file_size,
                    'write_time': write_time,
                    'read_time': read_time,
                    'verification': verification
                }
                
                status = "✓ VERIFIED" if verification.get('values_match', False) else "✗ FAILED"
                print(f"{compression}: {file_size/1024/1024:.2f} MB, "
                      f"Write: {write_time:.2f}s, Read: {read_time:.2f}s, {status}")
                
            except Exception as e:
                print(f"Error with {compression}: {e}")
                compression_results[compression] = None
                # Record verification failure
                verification = {'format': f"{model_name}_{compression}", 'error': str(e)}
                self.verification_results[f"{model_name}_{compression}"] = verification
        
        return compression_results
    
    def print_verification_report(self, model_name):
        """Minimal verification report - just return status"""
        failed_formats = []
        
        for format_name, verification in self.verification_results.items():
            if not format_name.startswith(model_name):
                continue
                
            if 'error' in verification and verification['error']:
                failed_formats.append(format_name)
            elif not verification.get('values_match', False):
                failed_formats.append(format_name)
        
        return len(failed_formats) == 0
    
    def analyze_and_report(self, baseline_sizes, parquet_results, model_name, embedding_dims):
        """Generate clean compression analysis report"""
        print(f"\nCOMPRESSION RESULTS - {model_name} ({embedding_dims}D)")
        print(f"{'Algorithm':<15} {'Size (MB)':<10} {'Ratio':<8} {'Reduction':<12} {'Verified':<10}")
        print("-" * 60)
        
        # Combine all formats into one table
        all_results = []
        
        # Add baseline formats
        for format_name, size in baseline_sizes.items():
            ratio = baseline_sizes['numpy'] / size
            reduction = (1 - size / baseline_sizes['numpy']) * 100
            
            # Check verification status
            verification_key = f"{model_name}_{format_name}"
            is_verified = self.verification_results.get(verification_key, {}).get('values_match', False)
            verified_status = "✓" if is_verified else "✗"
            
            all_results.append({
                'algorithm': format_name,
                'size': size,
                'ratio': ratio,
                'reduction': reduction,
                'verified': is_verified,
                'verified_status': verified_status
            })
        
        # Add parquet formats
        for algo, results in parquet_results.items():
            if results is None:
                continue
                
            ratio = baseline_sizes['numpy'] / results['file_size']
            reduction = (1 - results['file_size'] / baseline_sizes['numpy']) * 100
            is_verified = results.get('verification', {}).get('values_match', False)
            verified_status = "✓" if is_verified else "✗"
            
            all_results.append({
                'algorithm': algo,
                'size': results['file_size'],
                'ratio': ratio,
                'reduction': reduction,
                'verified': is_verified,
                'verified_status': verified_status
            })
        
        # Sort by size (smallest first)
        all_results.sort(key=lambda x: x['size'])
        
        # Print table
        for result in all_results:
            size_mb = result['size'] / 1024 / 1024
            print(f"{result['algorithm']:<15} {size_mb:<10.2f} {result['ratio']:<8.2f} "
                  f"{result['reduction']:<12.1f}% {result['verified_status']:<10}")
        
        # Find best verified compression
        verified_results = [r for r in all_results if r['verified']]
        if verified_results:
            best = min(verified_results, key=lambda x: x['size'])
            print(f"\nBest compression: {best['algorithm']} achieves {best['reduction']:.1f}% size reduction.")
        else:
            print(f"\n⚠️  No compression algorithm passed verification!")
            best = {'algorithm': 'NONE', 'ratio': 1.0, 'size': baseline_sizes['numpy']}
        
        return {
            'best_algorithm': best['algorithm'],
            'best_ratio': best.get('ratio', 1.0),
            'billion_size_gb': (best['size'] * 10000) / (1024**3),
            'baseline_sizes': baseline_sizes,
            'parquet_results': parquet_results
        }

def main():
    analyzer = EmbeddingCompressionAnalyzer()
    
    # Extract code snippets
    code_snippets = analyzer.extract_code_snippets(100000)
    print(f"Extracted {len(code_snippets)} code snippets")
    
    results = {}
    
    # Test both models
    for model_name, dims in analyzer.models.items():
        print(f"\nProcessing {model_name} ({dims} dimensions)...")
        
        # Generate embeddings
        embeddings = analyzer.generate_embeddings(code_snippets, model_name)
        
        # Save baseline formats with verification
        baseline_sizes = analyzer.save_baseline_formats(embeddings, model_name)
        
        # Test parquet compression with verification
        parquet_results = analyzer.test_parquet_compression(embeddings, model_name)
        
        # Check verification status (silent)
        model_verified = analyzer.print_verification_report(model_name)
        
        # Analyze and report
        analysis = analyzer.analyze_and_report(baseline_sizes, parquet_results, model_name, dims)
        results[model_name] = analysis

if __name__ == "__main__":
    main()