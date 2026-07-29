from pathlib import Path

# Set the root directory path
root_dir = Path("D:/Desktop/UoM/Academic/FYP/ayurveda-recognition/model_training/module_3/dataset/health_labelled")
# Set of valid image extensions to count
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Check if root directory exists
if not root_dir.exists():
    print(f"Directory '{root_dir}' not found.")
else:
    print(f"{'Folder Path':<60} | {'Image Count':<10}")
    print("-" * 75)

    # Walk through all directories starting from root
    for path in sorted(root_dir.rglob("*")):
        if path.is_dir():
            # Count images matching the extensions in the current folder
            image_count = sum(
                1 for file in path.iterdir() 
                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
            )
            
            # Print only folders that actually contain images
            if image_count > 0:
                print(f"{str(path):<60} | {image_count:<10}")