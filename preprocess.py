import os
import numpy as np
from PIL import Image

def extract_features_labels(dataset_path="dataset"):
    X, y = [], []
    for category in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, category)
        label = 1 if category.lower() == "positive" else 0
        print(f"Reading {folder} with label {label}")
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                img = Image.open(img_path).resize((64, 64)).convert("L")
                img_array = np.array(img).flatten() / 255.0
                X.append(img_array)
                y.append(label)
            except:
                pass
    return np.array(X), np.array(y)
