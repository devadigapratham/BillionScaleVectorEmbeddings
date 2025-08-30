#!/bin/bash

echo "Working Vector Search Benchmarking Framework"
echo "=============================================="
echo "Note: HNSWlib excluded due to system compatibility issues"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import numpy, pandas, matplotlib, seaborn, psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Some required packages are missing."
    echo "Please run: ./install_dependencies.sh"
    exit 1
fi

# Check if FAISS is available
python3 -c "import faiss" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: FAISS not available. Install with: pip install faiss-cpu"
    echo "This is required for the benchmarks to run."
    exit 1
fi

# Check if data directory exists
if [ ! -d "./million_embeddings" ]; then
    echo "ERROR: Data directory './million_embeddings' not found."
    echo "Please ensure your embeddings are in the correct directory structure."
    echo ""
    echo "Expected structure:"
    echo "  ./million_embeddings/"
    echo "  ├── dim_128/"
    echo "  ├── dim_256/"
    echo "  ├── dim_512/"
    echo "  ├── dim_1024/"
    echo "  └── dim_2048/"
    exit 1
fi

# Create results directory
mkdir -p ./results

echo ""
echo "Starting full benchmarks on million vector dataset..."
echo "Data directory: ./million_embeddings"
echo "Results directory: ./results"
echo "Libraries: FAISS, scikit-learn, SciPy"
echo ""

# Run working benchmarks
python3 working_benchmarks.py --data-dir ./million_embeddings --results-dir ./results

if [ $? -eq 0 ]; then
    echo ""
    echo "Full benchmarks completed successfully!"
    echo ""
    echo "Results saved to: ./results/"
    echo "Files generated:"
    echo "  - benchmark_results.csv - Raw performance data"
echo "  - benchmark_overview.png - Performance overview charts"
echo "  - summary_statistics.png - Summary statistics table"
echo "  - benchmark_report.txt - Comprehensive analysis report"
    echo ""
    echo "You can view the results in the ./results/ directory"
    echo ""
    echo "Key insights:"
    echo "  - FAISS is typically fastest for production use"
    echo "  - scikit-learn is good for research/development"
    echo "  - SciPy provides baseline performance"
    echo ""
    echo "Performance ranking (expected):"
    echo "  1. FAISS IndexFlatIP - Fastest queries"
    echo "  2. FAISS IndexIVFFlat - Balanced performance"
    echo "  3. scikit-learn BallTree - Good for medium datasets"
    echo "  4. SciPy cdist - Baseline (slowest but exact)"
else
    echo ""
    echo "ERROR: Benchmarks failed. Please check the error messages above."
    exit 1
fi
