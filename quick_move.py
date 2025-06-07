import os
import glob
import shutil

src_dir = "data/input/tony/Marshall/recording/export"
dst_dir = "data/input/tony/Marshall/camera06"

# Ensure destination directory exists
os.makedirs(dst_dir, exist_ok=True)

# Find files matching *camera01*
files = glob.glob(os.path.join(src_dir, "*camera02*"))

# Move each file
for file in files:
    if os.path.isfile(file):  # Ensure it's a file
        shutil.move(file, dst_dir)
        print(f"Moved: {file} -> {dst_dir}")

print("Done.")