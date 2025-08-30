#!/bin/bash

echo "Installing Vector Search Benchmarking Dependencies..."
echo "=================================================="

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y python3-pip python3-dev build-essential

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Install FAISS (CPU version)
echo "Installing FAISS CPU version..."
pip3 install faiss-cpu

echo ""
echo "Installation completed!"
echo "You can now run the benchmarks with: python3 benchmarks.py"
echo ""
echo "Available options:"
echo "  --data-dir: Directory containing vector embeddings (default: ./million_embeddings)"
echo "  --results-dir: Directory to save results (default: ./results)"
echo "  --dims: Specific dimensions to benchmark (default: all dimensions)"
echo ""
echo "Example: python3 benchmarks.py --data-dir ./million_embeddings --results-dir ./results"
