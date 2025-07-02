#!/bin/bash
# This script builds the project using the specified build tool.

# Step 1: Install pre-commit and set up hooks
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
fi
pre-commit install
pre-commit install --hook-type pre-push

# Step 2: Install the environment (add your logic here if needed)

# Step 3: Build the project
echo "Building the project..."
python -m build

# Step 4: Install the built wheel
echo "Installing the built wheel..."
pip install --upgrade dist/*.whl --force-reinstall
