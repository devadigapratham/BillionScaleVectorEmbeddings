
# Billion Scale Vector Embeddings

This repository is part of my Google Summer of Code (GSoC) project under the UC OSPO organization. It provides tools for downloading open-source repositories, generating large-scale vector embeddings using transformer models, and benchmarking vector search performance across multiple libraries and datasets.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarking](#benchmarking)
- [Code Structure](#code-structure)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

The project is designed to scale up vector embedding generation using models like CodeBERT on hundreds of open-source repositories, targeting up to 1 billion embeddings. It supports efficient chunking, batching, GPU-accelerated inference, and compression for embedding storage. The project also includes a comprehensive benchmarking framework for evaluating vector search performance across different libraries and datasets.

## Features

- **Large-scale embedding generation**: Process millions of code files and generate embeddings
- **Multiple transformer models**: Support for various pre-trained models including CodeBERT
- **Efficient storage**: Compressed embedding storage with pickle and LZMA compression
- **GPU acceleration**: CUDA support for faster inference
- **Comprehensive benchmarking**: Test vector search performance across multiple libraries
- **Scalable architecture**: Designed to handle billion-scale datasets
- **Multiple dimensions**: Support for various embedding dimensions (128, 256, 512, 1024, 2048)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)
- At least 16GB RAM (recommended for large datasets)

### Quick Install

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/BillionScaleVectorEmbeddings.git
    cd BillionScaleVectorEmbeddings
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. For benchmarking capabilities, install additional dependencies:
    ```bash
    cd benchmarking
    chmod +x install_dependencies.sh
    ./install_dependencies.sh
    ```

### Manual Installation

If you prefer manual installation:

```bash
# Core dependencies
pip install numpy pandas pyarrow sentence-transformers torch modal-client pynvml

# Benchmarking dependencies
pip install matplotlib seaborn psutil faiss-cpu scipy scikit-learn hnswlib
```

## Usage

### 1. Clone repositories

Downloads and stores the list of target repositories inside the `repositories/` directory:

```bash
python cloneRepos.py
```

### 2. Generate vector embeddings

Processes all files, splits them into overlapping chunks, and generates embeddings using the selected transformer model. Embeddings are saved in compressed shards:

```bash
python generateEmbeddings.py
```

**Configuration options:**
- Model selection (default: CodeBERT)
- Chunk size and overlap
- Batch size for processing
- Output directory and compression settings

### 3. Verify saved embeddings

Verifies that all saved embedding files have the correct number of vectors and expected dimensions:

```bash
python verifyEmbeddings.py
```

### 4. Run benchmarks

Evaluate vector search performance across different libraries and datasets:

```bash
cd benchmarking
python benchmarks.py
```

## Benchmarking

The project includes a comprehensive benchmarking framework for evaluating vector search performance. See the [benchmarking README](benchmarking/README.md) for detailed information.

### Supported Libraries

- **FAISS**: High-performance similarity search and clustering
- **HNSWlib**: Hierarchical Navigable Small World graphs
- **scikit-learn**: Machine learning algorithms including BallTree
- **SciPy**: Scientific computing with brute force search

### Benchmark Features

- Performance metrics: build time, query time, memory usage, throughput
- Multiple embedding dimensions: 128, 256, 512, 1024, 2048
- Comprehensive reporting with charts and analysis
- Scalable testing on million+ vector datasets

### Quick Benchmark

```bash
cd benchmarking
python benchmarks.py --data-dir ./your_embeddings --results-dir ./results
```

## Code Structure

```
BillionScaleVectorEmbeddings/
├── cloneRepos.py              # Repository cloning and management
├── generateEmbeddings.py      # Main embedding generation pipeline
├── verifyEmbeddings.py        # Embedding validation and verification
├── requirements.txt           # Core Python dependencies
├── benchmarking/              # Vector search benchmarking framework
│   ├── benchmarks.py         # Main benchmarking script
│   ├── install_dependencies.sh # Dependency installation script
│   ├── requirements.txt      # Benchmarking dependencies
│   └── README.md            # Detailed benchmarking documentation
├── embeddingsGenerationScripts/ # Legacy scripts and utilities
└── README.md                 # This file
```


## Data Format

The system generates embeddings in the following structure:

```
embeddings_directory/
├── dim_128/
│   ├── batch_000000.pkl.xz
│   ├── batch_000001.pkl.xz
│   └── ...
├── dim_256/
│   ├── batch_000000.pkl.xz
│   ├── batch_000001.pkl.xz
│   └── ...
└── ...
```

Each batch file contains a compressed numpy array with shape `(batch_size, dimension)`.

## Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Specify GPU devices to use
- `MODAL_TOKEN_ID`: Modal API token for cloud processing
- `MODAL_TOKEN_SECRET`: Modal API secret for cloud processing

### Model Configuration

- **Default model**: `microsoft/codebert-base`
- **Alternative models**: Any HuggingFace sentence transformer
- **Chunk size**: Configurable text chunking for code files
- **Overlap**: Configurable overlap between chunks

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or use smaller datasets
2. **CUDA errors**: Ensure compatible GPU drivers and PyTorch installation
3. **Slow performance**: Check if GPU is being utilized correctly
4. **Storage issues**: Ensure sufficient disk space for embeddings

### Performance Tips

- Use SSD storage for faster I/O operations
- Ensure sufficient RAM (2x dataset size recommended)
- Use GPU acceleration when available
- Optimize batch sizes for your hardware


## Citation

If you use this project in your research, please cite:

```bibtex
@software{billion_scale_vector_embeddings,
  title={Billion Scale Vector Embeddings},
  author={Prathamesh Devadiga, Jayjeet Chakraborty },
  year={2025},
  url={https://github.com/devadigapratham/BillionScaleVectorEmbeddings}
}
```

## Acknowledgments

- UC OSPO for GSoC support
- HuggingFace for transformer models
- FAISS team for vector search library
- Open source community for repositories used in testing
