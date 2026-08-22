from dataset import load_datasets

from model import build_model

train_ds, _, _ = load_datasets()

num_classes = len(train_ds.class_names)

model = build_model(num_classes)

model.summary()