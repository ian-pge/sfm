
from pathlib import Path
import os
import shutil

# Setup dummy dataset
dataset_path = Path("test_dataset")
if dataset_path.exists():
    shutil.rmtree(dataset_path)
images_path = dataset_path / "images"
images_path.mkdir(parents=True)

# Create 5 dummy images
image_names = ["img_1.jpg", "img_2.jpg", "img_3.jpg", "img_4.jpg", "img_5.jpg"]
for name in image_names:
    (images_path / name).touch()

# Import the function from pipeline.py
import sys
sys.path.append("scripts")
from pipeline import generate_sequential_pairs

output_path = Path("test_output")
output_path.mkdir(exist_ok=True)
pairs_path = output_path / "pairs.txt"

# Run generation with window size 2
generate_sequential_pairs(images_path, pairs_path, window_size=2)

# Verify content
with open(pairs_path, "r") as f:
    lines = f.readlines()
    print("Generated pairs:")
    for line in lines:
        print(line.strip())

expected = [
    "img_1.jpg img_2.jpg",
    "img_1.jpg img_3.jpg",
    "img_2.jpg img_3.jpg",
    "img_2.jpg img_4.jpg",
    "img_3.jpg img_4.jpg",
    "img_3.jpg img_5.jpg",
    "img_4.jpg img_5.jpg"
]

print("-" * 20)
pairs_set = set(line.strip() for line in lines)
expected_set = set(expected)

missing = expected_set - pairs_set
extra = pairs_set - expected_set

if not missing and not extra:
    print("SUCCESS: Pairs match expected sequential pattern.")
else:
    print("FAILURE: Pairs do not match expected.")
    if missing:
        print(f"Missing: {missing}")
    if extra:
        print(f"Extra: {extra}")

# Cleanup
# shutil.rmtree(dataset_path)
# shutil.rmtree(output_path)
