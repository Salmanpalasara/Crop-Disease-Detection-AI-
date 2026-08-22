from pathlib import Path
from collections import Counter
from PIL import Image
from tqdm import tqdm

from config import TRAIN_DIR

print("Train Path:", TRAIN_DIR)
print("Exists:", TRAIN_DIR.exists())

print("=" * 60)
print("CROP DISEASE DATASET ANALYSIS")
print("=" * 60)

classes = sorted(
    [
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    ]
)

print(f"\nTotal Classes : {len(classes)}\n")

image_counter = Counter()

total_images = 0

corrupted = []

extensions = Counter()

for cls in classes:

    folder = TRAIN_DIR / cls

    images = list(folder.glob("*"))

    image_counter[cls] = len(images)

    total_images += len(images)

    for img_path in tqdm(images, desc=cls):

        try:

            img = Image.open(img_path)

            img.verify()

            extensions[img_path.suffix.lower()] += 1

        except Exception:

            corrupted.append(str(img_path))

print("\n")
print("=" * 60)
print(f"Total Images : {total_images}")
print("=" * 60)

print("\nImages Per Class\n")

for cls, count in image_counter.items():

    print(f"{cls:<45} {count}")

largest = image_counter.most_common(1)[0]

smallest = image_counter.most_common()[-1]

print("\nLargest Class")
print(largest)

print("\nSmallest Class")
print(smallest)

print("\nImage Formats")

for ext, count in extensions.items():

    print(ext, count)

print("\nCorrupted Images :", len(corrupted))

if corrupted:

    print("\nCorrupted Files")

    for file in corrupted:

        print(file)

print("\nAnalysis Completed Successfully")