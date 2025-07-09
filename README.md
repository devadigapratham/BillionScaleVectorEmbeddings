
# Billion Scale Vector Embeddings

This repository is part of my Google Summer of Code (GSoC) project under the UC OSPO organization. It provides tools for downloading open-source repositories, generating large-scale vector embeddings using transformer models, and verifying the generated embedding shards.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)

## Overview

The project is designed to scale up vector embedding generation using models like CodeBERT on hundreds of open-source repositories, targeting up to 1 billion embeddings. It supports efficient chunking, batching, GPU-accelerated inference, and compression for embedding storage.

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

### 1. Clone repositories
Downloads and stores the list of target repositories inside the `repositories/` directory:
```bash
python cloneRepos.py
````

### 2. Generate vector embeddings

Processes all files, splits them into overlapping chunks, and generates embeddings using the selected transformer model. Embeddings are saved in compressed shards inside `embeddings_billion/`:

```bash
python generateEmbeddings.py
```

### 3. Verify saved embeddings

Verifies that all saved `.pkl.xz` embedding files have the correct number of vectors and expected dimensions:

```bash
python verifyEmbeddings.py
```

## Code Structure

* `cloneRepos.py` – Clones large open-source repositories (e.g., TensorFlow, PyTorch).
* `generateEmbeddings.py` – Generates embeddings using a transformer model, stores shards efficiently.
* `verifyEmbeddings.py` – Validates dimensions and counts of stored embedding shards.
* `requirements.txt` – Python package dependencies.
* `oldfiles/` – Backup directory for legacy scripts and experiments.
