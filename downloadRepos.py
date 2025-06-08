import os 
import subprocess
import sys
from pathlib import Path
import time
from collections import defaultdict
import re

# List of repositories to analyze
REPOSITORIES = [
    {
        'name': 'Linux Kernel',
        'url': 'https://github.com/torvalds/linux.git',
        'branch': 'v6.12',  # Specific version
        'dir': 'linux-kernel'
    },
    {
        'name': 'LLVM Project', 
        'url': 'https://github.com/llvm/llvm-project.git',
        'branch': 'main',
        'dir': 'llvm-project'
    },
    {
        'name': 'QEMU',
        'url': 'https://github.com/qemu/qemu.git', 
        'branch': 'master',
        'dir': 'qemu'
    },
    {
        'name': 'Systemd',
        'url': 'https://github.com/systemd/systemd.git',
        'branch': 'main', 
        'dir': 'systemd'
    },
    {
        'name': 'GCC',
        'url': 'https://github.com/gcc-mirror/gcc.git',
        'branch': 'master',
        'dir': 'gcc'
    },
    {
        'name': 'Chromium',
        'url': 'https://chromium.googlesource.com/chromium/src.git',
        'branch': 'main',
        'dir': 'chromium-src'
    },
    {
        'name': 'Android AOSP',
        'url': 'https://android.googlesource.com/platform/frameworks/base.git',
        'branch': 'master', 
        'dir': 'android-frameworks-base'
    },
    {
        'name': 'FreeBSD',
        'url': 'https://github.com/freebsd/freebsd-src.git',
        'branch': 'main',
        'dir': 'freebsd-src'
    }
]

# File extensions to include for token counting
CODE_EXTENSIONS = {
    '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx',  # C/C++
    '.py', '.pyx',  # Python
    '.java', '.kt',  # Java/Kotlin
    '.js', '.ts', '.jsx', '.tsx',  # JavaScript/TypeScript
    '.go',  # Go
    '.rs',  # Rust
    '.sh', '.bash', '.zsh',  # Shell
    '.pl', '.pm',  # Perl
    '.rb',  # Ruby
    '.php',  # PHP
    '.swift',  # Swift
    '.m', '.mm',  # Objective-C
    '.asm', '.s', '.S',  # Assembly
    '.cmake', '.mk', '.make',  # Build files
    '.xml', '.json', '.yaml', '.yml',  # Config files
    '.md', '.rst', '.txt',  # Documentation
    '.sql',  # SQL
    '.proto',  # Protocol Buffers
    '.thrift',  # Thrift
    '.capnp'  # Cap'n Proto
}

def run_command(cmd, cwd=None, timeout=3600):
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, 
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {cmd}")
        return False, "", "Timeout"

def clone_repository(repo):
    print(f"\n{'='*50}")
    print(f"Cloning {repo['name']}...")
    print(f"URL: {repo['url']}")
    print(f"Branch: {repo['branch']}")
    
    # Remove existing directory if it exists
    if os.path.exists(repo['dir']):
        print(f"Removing existing directory: {repo['dir']}")
        subprocess.run(f"rm -rf {repo['dir']}", shell=True)
    
    # Clone with depth=1 for faster cloning (only latest commit)
    clone_cmd = f"git clone --depth 1 --branch {repo['branch']} {repo['url']} {repo['dir']}"
    
    start_time = time.time()
    success, stdout, stderr = run_command(clone_cmd)
    clone_time = time.time() - start_time
    
    if success:
        print(f"✓ Cloned successfully in {clone_time:.1f}s")
        return True
    else:
        print(f"✗ Clone failed: {stderr}")
        return False

def estimate_tokens_simple(text):
    """Simple token estimation: ~1 token per 4 characters"""
    return len(text) // 4

def estimate_tokens_word_based(text):
    """Word-based token estimation: ~0.75 tokens per word"""
    words = len(re.findall(r'\b\w+\b', text))
    return int(words * 0.75)

def count_file_tokens(file_path):
    """Count tokens in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            return estimate_tokens_simple(content), len(content)
    except Exception as e:
        return 0, 0

def analyze_repository(repo):
    """Analyze repository and count tokens"""
    print(f"\nAnalyzing {repo['name']}...")
    
    if not os.path.exists(repo['dir']):
        print(f"Directory {repo['dir']} not found!")
        return None
    
    stats = {
        'name': repo['name'],
        'total_files': 0,
        'code_files': 0,
        'total_tokens': 0,
        'total_chars': 0,
        'file_types': defaultdict(int),
        'largest_files': []
    }
    
    # Walk through all files
    for root, dirs, files in os.walk(repo['dir']):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
            
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file).suffix.lower()
            
            stats['total_files'] += 1
            stats['file_types'][file_ext] += 1
            
            if file_ext in CODE_EXTENSIONS:
                stats['code_files'] += 1
                tokens, chars = count_file_tokens(file_path)
                stats['total_tokens'] += tokens
                stats['total_chars'] += chars
                
                # Track largest files
                if tokens > 1000:  # Only track files with >1000 tokens
                    stats['largest_files'].append((file_path, tokens))
    
    # Sort largest files
    stats['largest_files'].sort(key=lambda x: x[1], reverse=True)
    stats['largest_files'] = stats['largest_files'][:10]  # Top 10
    
    return stats

def format_number(num):
    """Format large numbers with commas"""
    return f"{num:,}"

def calculate_embedding_size(tokens, dimensions):
    """Calculate storage size for embeddings"""
    # Assuming float32 (4 bytes per dimension)
    bytes_per_embedding = dimensions * 4
    total_bytes = tokens * bytes_per_embedding
    
    # Convert to human readable
    if total_bytes > 1e12:  # TB
        return f"{total_bytes/1e12:.2f} TB"
    elif total_bytes > 1e9:  # GB
        return f"{total_bytes/1e9:.2f} GB"
    elif total_bytes > 1e6:  # MB
        return f"{total_bytes/1e6:.2f} MB"
    else:
        return f"{total_bytes/1e3:.2f} KB"

def print_results(all_stats):
    """Print comprehensive results"""
    print(f"\n{'='*80}")
    print("VECTOR EMBEDDING DATASET ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    total_tokens = 0
    total_files = 0
    
    for stats in all_stats:
        if stats is None:
            continue
            
        print(f"\n📁 {stats['name']}")
        print(f"   Total files: {format_number(stats['total_files'])}")
        print(f"   Code files: {format_number(stats['code_files'])}")
        print(f"   Total tokens: {format_number(stats['total_tokens'])}")
        print(f"   Total characters: {format_number(stats['total_chars'])}")
        
        # Embedding sizes
        size_384 = calculate_embedding_size(stats['total_tokens'], 384)
        size_768 = calculate_embedding_size(stats['total_tokens'], 768)
        print(f"   Embedding size (384D): {size_384}")
        print(f"   Embedding size (768D): {size_768}")
        
        # Top file types
        top_types = sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   Top file types: {', '.join([f'{ext}({count})' for ext, count in top_types])}")
        
        total_tokens += stats['total_tokens']
        total_files += stats['code_files']
    
    print(f"\n{'='*80}")
    print("TOTAL DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Total code files: {format_number(total_files)}")
    print(f"Total tokens: {format_number(total_tokens)}")
    print(f"Total embedding storage (384D): {calculate_embedding_size(total_tokens, 384)}")
    print(f"Total embedding storage (768D): {calculate_embedding_size(total_tokens, 768)}")
    
    # Scale estimation
    billion_scale = total_tokens / 1e9
    print(f"Scale: {billion_scale:.2f} billion tokens")
    

def main():
    """Main execution function"""
    print("Vector Embedding Dataset Builder")
    print("Targeting billion-scale token count for embedding generation")
    print(f"Models: all-MiniLM-L6-v2 (384D) & all-mpnet-base-v2 (768D)")
    
    # Create working directory
    work_dir = "embedding_datasets"
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    
    all_stats = []
    
    # Process each repository
    for repo in REPOSITORIES:
        try:
            # Clone repository
            if clone_repository(repo):
                # Analyze repository
                stats = analyze_repository(repo)
                all_stats.append(stats)
            else:
                print(f"Skipping analysis for {repo['name']} due to clone failure")
                all_stats.append(None)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"Error processing {repo['name']}: {e}")
            all_stats.append(None)
    
    # Print final results
    print_results(all_stats)
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()