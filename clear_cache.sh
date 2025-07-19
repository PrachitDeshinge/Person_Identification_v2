#!/bin/bash

clear

# Clear Python cache files and directories
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# Clear Dstore files (commonly used for Fairseq or similar projects)
find . -type f -name "*.dstore*" -delete

echo "Cache and Dstore files cleared."