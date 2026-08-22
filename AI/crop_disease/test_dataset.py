from dataset import load_datasets
from dataset import get_class_names
from dataset import calculate_class_weights

train_ds, val_ds, test_ds = load_datasets()

print("=" * 60)

print("Training Dataset")

print(train_ds)

print("=" * 60)

print("Validation Dataset")

print(val_ds)

print("=" * 60)

print("Test Dataset")

print(test_ds)

print("=" * 60)

classes = get_class_names(train_ds)

print(f"\nTotal Classes : {len(classes)}\n")

for cls in classes:

    print(cls)

print("\n")

weights = calculate_class_weights(train_ds)

print(weights)