# Vector Search Benchmarking Framework

A comprehensive benchmarking framework for evaluating vector search libraries and indexes on large-scale embedding datasets.

## Features

- **Multiple Libraries**: Benchmarks FAISS, HNSWlib, scikit-learn, and SciPy
- **Various Index Types**: Tests different indexing strategies (exact search, approximate search)
- **Performance Metrics**: Measures build time, query time, memory usage, and throughput
- **Comprehensive Analysis**: Generates tables, graphs, and detailed reports
- **Scalable**: Designed to handle million+ vector embeddings across multiple dimensions

## Supported Libraries

| Library | Index Types | Use Case |
|---------|-------------|----------|
| **FAISS** | IndexFlatIP, IndexIVFFlat | Production, high-performance |
| **HNSWlib** | HNSW (Hierarchical Navigable Small World) | Production, approximate search |
| **scikit-learn** | BallTree | Research, development |
| **SciPy** | cdist (brute force) | Baseline, small datasets |

## Installation

### Quick Install (Linux/Ubuntu)

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

### Manual Install

1. **Install system dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip python3-dev build-essential
   ```

2. **Install Python packages:**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Install FAISS (CPU version):**
   ```bash
   pip3 install faiss-cpu
   ```

## Usage

### Basic Usage

```bash
python3 benchmarks.py
```

This will:
- Load embeddings from `./million_embeddings/`
- Run benchmarks across all dimensions (128, 256, 512, 1024, 2048)
- Save results to `./results/`

### Advanced Usage

```bash
# Specify custom data and results directories
python3 benchmarks.py --data-dir ./my_embeddings --results-dir ./my_results

# Benchmark specific dimensions only
python3 benchmarks.py --dims 128 256 512

# Combine options
python3 benchmarks.py --data-dir ./million_embeddings --results-dir ./results --dims 128 256
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir` | Directory containing vector embeddings | `./million_embeddings` |
| `--results-dir` | Directory to save benchmark results | `./results` |
| `--dims` | Specific dimensions to benchmark | All dimensions |

## Data Format

The framework expects embeddings organized in the following structure:

```
data_directory/
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

Each batch file should contain a numpy array of shape `(batch_size, dimension)`.

## Output

The framework generates comprehensive results in the specified results directory:

### Files Generated

1. **`benchmark_results.csv`** - Raw benchmark data in CSV format
2. **`benchmark_overview.png`** - Overview charts (build time, query time, throughput, memory)
3. **`detailed_analysis.png`** - Detailed performance analysis with heatmaps
4. **`summary_statistics.png`** - Summary statistics table
5. **`benchmark_report.txt`** - Comprehensive text report

### Metrics Measured

- **Build Time**: Time to construct the index
- **Query Time**: Average time for nearest neighbor search
- **Memory Usage**: Memory consumed by the index
- **Throughput**: Queries per second (QPS)
- **Recall**: Accuracy of search results (100% for exact search)

## Performance Considerations

### Memory Usage
- **FAISS**: Efficient memory usage, scales well with dataset size
- **HNSWlib**: Moderate memory usage, good for approximate search
- **scikit-learn**: Higher memory usage, suitable for smaller datasets
- **SciPy**: Highest memory usage, loads entire dataset into memory

### Query Performance
- **FAISS**: Fastest query times, optimized for large-scale search
- **HNSWlib**: Fast approximate search with configurable accuracy
- **scikit-learn**: Good performance for medium-sized datasets
- **SciPy**: Slowest, but provides exact results

### Build Time
- **FAISS IndexFlatIP**: Fastest build time
- **FAISS IndexIVFFlat**: Moderate build time (includes training)
- **HNSWlib**: Moderate build time
- **scikit-learn**: Fast build time
- **SciPy**: No build time (brute force)

## Example Results

After running benchmarks, you'll get:

```
results/
├── benchmark_results.csv          # Raw data
├── benchmark_overview.png         # Performance overview
├── detailed_analysis.png          # Detailed analysis
├── summary_statistics.png         # Summary table
└── benchmark_report.txt           # Text report
```

## Customization

### Adding New Libraries

To add a new vector search library:

1. Add the library to the imports section
2. Create a new benchmark method (e.g., `benchmark_newlib`)
3. Add the benchmark call in `run_benchmarks()`
4. Update the `BenchmarkResult` dataclass if needed

### Modifying Metrics

The framework measures standard metrics, but you can extend it by:

- Adding new fields to `BenchmarkResult`
- Implementing custom measurement functions
- Adding new visualization types

## Troubleshooting

### Common Issues

1. **FAISS not available**: Install with `pip install faiss-cpu`
2. **Memory errors**: Reduce batch size or use fewer dimensions
3. **Slow performance**: Ensure you have sufficient RAM and CPU cores

### Performance Tips

- Use SSD storage for faster data loading
- Ensure sufficient RAM (recommend 2x dataset size)
- Run benchmarks during low system load
- Consider benchmarking subsets first

## Contributing

Contributions are welcome! Areas for improvement:

- Additional vector search libraries
- New performance metrics
- Enhanced visualizations
- GPU acceleration support
- Distributed benchmarking

## License

This project is open source. Feel free to use and modify as needed.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{vector_search_benchmarks,
  title={Vector Search Benchmarking Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/vector-search-benchmarks}
}
```
