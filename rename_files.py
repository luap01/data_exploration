import os
from pathlib import Path

def rename_files():
    # Directory containing the files
    directory = Path("data/input/openpose/images/camera05")
    
    # Get all jpg files
    files = list(directory.glob("*.jpg"))
    
    # Sort files to ensure consistent ordering
    files.sort()
    
    # Rename each file
    for file in files:
        # Get the number part (remove .jpg and leading zeros)
        num = int(file.stem)
        # Create new filename with 6 digits
        new_name = f"{num:06d}.jpg"
        # Create full path for new name
        new_path = directory / new_name
        
        # Only rename if the new name is different
        if file.name != new_name:
            print(f"Renaming {file.name} to {new_name}")
            os.rename(file, new_path)

if __name__ == "__main__":
    rename_files() 