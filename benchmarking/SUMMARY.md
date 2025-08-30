# Vector Search Benchmarking Framework - Complete Setup

## What We've Built

I've created a comprehensive vector search benchmarking framework that will test your million vector embeddings dataset across multiple libraries and provide detailed performance analysis with tables, graphs, and reports.

## Files Created

### Core Framework
- **`benchmarks.py`** - Main benchmarking script with comprehensive testing
- **`benchmark_config.py`** - Configuration file for customizing parameters
- **`requirements.txt`** - Python dependencies

### Scripts & Utilities
- **`install_dependencies.sh`** - Automated dependency installation
- **`run_benchmarks.sh`** - Easy-to-use benchmark runner
- **`test_benchmarks.py`** - Test script to verify setup

### Documentation
- **`README.md`** - Comprehensive usage guide
- **`SUMMARY.md`** - This summary file

## What It Benchmarks

### Vector Search Libraries
1. **FAISS** (Facebook AI Similarity Search)
   - IndexFlatIP: Exact inner product search
   - IndexIVFFlat: Inverted file with flat compression

2. **HNSWlib** (Hierarchical Navigable Small World)
   - Approximate nearest neighbor search
   - Configurable accuracy vs. speed trade-offs

3. **scikit-learn**
   - BallTree: Space-partitioning tree structure
   - Good for research and development

4. **SciPy**
   - cdist: Brute force distance calculation
   - Baseline for exact search performance

### Performance Metrics
- **Build Time**: Index construction time
- **Query Time**: Nearest neighbor search performance
- **Memory Usage**: RAM consumption
- **Throughput**: Queries per second (QPS)
- **Recall**: Search accuracy (100% for exact search)

## Quick Start

### 1. Install Dependencies
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

### 2. Run Benchmarks
```bash
chmod +x run_benchmarks.sh
./run_benchmarks.sh
```

### 3. Or Run Manually
```bash
python3 benchmarks.py --data-dir ./million_embeddings --results-dir ./results
```

## What You'll Get

After running the benchmarks, you'll have a `./results/` folder containing:

### Data Files
- **`benchmark_results.csv`** - Raw performance data
- **`benchmark_report.txt`** - Comprehensive analysis

### Visualizations
- **`benchmark_overview.png`** - Performance overview charts
- **`detailed_analysis.png`** - Detailed performance heatmaps
- **`summary_statistics.png`** - Summary statistics table

## Customization Options

### Command Line Arguments
- `--data-dir`: Specify your embeddings directory
- `--results-dir`: Choose where to save results
- `--dims`: Test specific dimensions only

### Configuration File
Edit `benchmark_config.py` to:
- Adjust number of queries
- Enable/disable specific libraries
- Customize visualization settings
- Set performance thresholds

## Expected Results

Based on typical performance characteristics:

### Performance Ranking (Expected)
1. **FAISS IndexFlatIP** - Fastest queries, moderate memory
2. **HNSWlib** - Fast approximate search, good memory efficiency
3. **FAISS IndexIVFFlat** - Balanced performance, good for large datasets
4. **scikit-learn BallTree** - Good for medium datasets
5. **SciPy cdist** - Slowest but exact results

### Memory Usage (Per Million Vectors)
- **FAISS**: 100-500 MB
- **HNSWlib**: 200-800 MB
- **scikit-learn**: 500-2000 MB
- **SciPy**: 2000-10000 MB

## Testing the Setup

Before running on your full dataset, test with:
```bash
python3 test_benchmarks.py
```

This creates a small test dataset and verifies everything works.

## Pro Tips

1. **Start Small**: Test with 1-2 dimensions first
2. **Monitor Memory**: Ensure you have enough RAM (2x dataset size recommended)
3. **Use SSD**: Faster data loading from solid-state storage
4. **Run During Off-Peak**: Avoid system load for consistent results
5. **Check Dependencies**: Ensure all libraries are properly installed

## Troubleshooting

### Common Issues
- **FAISS not found**: Run `pip install faiss-cpu`
- **Memory errors**: Reduce batch size or test fewer dimensions
- **Slow performance**: Check system resources and storage type

### Performance Tips
- Use sufficient RAM (recommend 2x dataset size)
- Ensure SSD storage for faster loading
- Run during low system load
- Consider benchmarking subsets first

## What's Next

After running the benchmarks:

1. **Analyze Results**: Review the generated charts and reports
2. **Compare Libraries**: See which performs best for your use case
3. **Optimize**: Use the best-performing library for your production system
4. **Scale**: Apply insights to larger datasets

## Understanding the Results

### Build Time
- Lower is better
- FAISS IndexFlatIP should be fastest
- HNSWlib and IVFFlat include training time

### Query Time
- Lower is better
- FAISS should be fastest
- HNSWlib provides speed-accuracy trade-offs

### Throughput (QPS)
- Higher is better
- Shows queries per second
- Key metric for production systems

### Memory Usage
- Lower is better
- Consider your system constraints
- FAISS is most memory-efficient

## Support

If you encounter issues:
1. Check the error messages
2. Verify all dependencies are installed
3. Test with the test script first
4. Check system resources (RAM, storage)

---

**Ready to benchmark your million vector embeddings? Run `./run_benchmarks.sh` and let the analysis begin!**
