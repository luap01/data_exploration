#!/bin/bash

# Check if folder parameter is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a folder name"
    echo "Usage: $0 <folder_name>"
    exit 1
fi

FOLDER=$1

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Cleaning openpose output for camera0$FOLDER"
rm -rf "$SCRIPT_DIR/../data/output/openpose/images/camera0$FOLDER"/*
rm -rf "$SCRIPT_DIR/../data/output/openpose/json/camera0$FOLDER"/*