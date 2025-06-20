# BillionScaleVectorEmbeddings

This repository is part of my Google Summer of Code (GSoC) project under the UC OSPO organization. It provides tools for downloading repositories, generating and comparing vector embeddings, and benchmarking GPU performance for large-scale vector operations.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)

## Overview

The project is designed to handle large-scale vector embeddings, including downloading datasets, comparing embedding methods, and measuring GPU computation times.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/BillionScaleVectorEmbeddings.git
    cd BillionScaleVectorEmbeddings
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the following scripts in order:

1. **Download repositories:**
    ```bash
    python downloadRepos.py
    ```
    This script downloads the required repositories or datasets.

2. **Generate and compare embeddings:**
    ```bash
    python embedding_comparision.py
    ```
    This script generates vector embeddings and compares different methods.

3. **Benchmark GPU time:**
    - For local GPU:
      ```bash
      python local_gpu_time.py
      ```
    - For modal GPU:
      ```bash
      python modal_gpu_time.py
      ```

4. **Generate embeddings on Multiple GPUs:** 
    - ```bash
        python modal_embeddings_generator.py
      ```

## Code Structure

- `downloadRepos.py` - Downloads datasets or repositories.
- `embedding_comparision.py` - Generates and compares vector embeddings.
- `local_gpu_time.py` - Benchmarks GPU performance locally.
- `modal_gpu_time.py` - Benchmarks GPU performance using Modal (uses only gcc repo for benchmarking)
- `modal_embeddings_generator` - Script for generating embeddings using multiple gpus on modal cloud. 

