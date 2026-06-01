import tensorflow as tf

model = tf.keras.models.load_model("models/repaired_model.h5")
print("✅ Model loaded successfully")
