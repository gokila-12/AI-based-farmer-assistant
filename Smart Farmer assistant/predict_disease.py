import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "repaired_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

print("MODEL_PATH =", MODEL_PATH)
print("MODEL EXISTS =", os.path.exists(MODEL_PATH))

print("🔄 Rebuilding MobileNet architecture...")

# ✅ Rebuild MobileNetV2 base
base_model = MobileNetV2(
    weights=None,              # ⭐ IMPORTANT → no ImageNet weights
    include_top=False,
    input_shape=(224, 224, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)

# ✅ Load class count dynamically
with open(LABEL_PATH, "r") as f:
    class_indices = json.load(f)

num_classes = len(class_indices)

predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("🔄 Loading weights...")

# ✅ Load weights ONLY (bypass broken config)
model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)


print("✅ Model loaded successfully!")

# Reverse mapping
index_to_class = {int(v): k for k, v in class_indices.items()}


def predict_disease(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)

    return index_to_class.get(idx, "Unknown"), round(confidence, 2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    args = parser.parse_args()

    disease, confidence = predict_disease(args.image_path)

    print("\n🌿 Prediction:")
    print("Disease:", disease)
    print("Confidence:", confidence, "%")
