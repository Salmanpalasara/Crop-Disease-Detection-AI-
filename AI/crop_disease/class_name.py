from dataset import load_datasets

train_ds, _, _ = load_datasets()

for i, name in enumerate(train_ds.class_names):
    print(i, name)