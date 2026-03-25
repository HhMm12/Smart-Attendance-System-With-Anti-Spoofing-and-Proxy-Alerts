import os
import cv2
import numpy as np
import subprocess
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── CONFIG ──
DATASET_PATH = "/Users/hrithikmedhi/Downloads/Minor Project Files/Dataset/LCC_FASD"
MODEL_H5 = "models/spoof_best.h5"
SAVED_MODEL = "models/spoof_saved"
MODEL_ONNX = "models/spoof_classifier.onnx"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20

# ── LOAD DATASET ──
print("📂 Loading dataset...")

images = []
labels = []

files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".png")]
total = len(files)

for i, fname in enumerate(files):
    if i % 1000 == 0:
        print(f"   Loading {i}/{total}...")

    label = 1 if fname.startswith("spoof_") else 0  # 1=spoof, 0=real

    path = os.path.join(DATASET_PATH, fname)
    img = cv2.imread(path)
    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
    labels.append(label)

images = np.array(images, dtype=np.float32) / 255.0
labels = np.array(labels)

print(f"✅ Loaded {len(images)} images")
print(f"   Real: {np.sum(labels==0)} | Spoof: {np.sum(labels==1)}")

# ── TRAIN/TEST SPLIT ──
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")


# ── BUILD MODEL ──
def build_model():
    model = models.Sequential(
        [
            # Block 1
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Classifier head
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax"),
        ]
    )
    return model


model = build_model()
model.summary()

# ── CLASS WEIGHTS ──
total_samples = len(y_train)
n_real = np.sum(y_train == 0)
n_spoof = np.sum(y_train == 1)
class_weight = {0: total_samples / (2 * n_real), 1: total_samples / (2 * n_spoof)}
print(f"   Class weights: real={class_weight[0]:.3f}, spoof={class_weight[1]:.3f}")

# ── COMPILE ──
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ── TRAIN ──
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_H5, save_best_only=True, verbose=1),
]

print("\n🚀 Training started...")
history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

# ── EVALUATE ──
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

# ── EXPORT TO ONNX ──
print("\n📦 Exporting to ONNX...")

# Step 1 — Load best saved weights
print("   Loading best model weights...")
best_model = tf.keras.models.load_model(MODEL_H5)

# Step 2 — Export as TF SavedModel format
print("   Saving as TF SavedModel...")
best_model.export(SAVED_MODEL)

# Step 3 — Convert SavedModel to ONNX using CLI
print("   Converting to ONNX...")
result = subprocess.run(
    [
        "python3",
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        SAVED_MODEL,
        "--output",
        MODEL_ONNX,
        "--opset",
        "13",
    ],
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.returncode == 0:
    print(f"✅ ONNX model saved to {MODEL_ONNX}")
else:
    print("❌ ONNX conversion failed:")
    print(result.stderr)
    print("\n⚠️  H5 model is still saved at:", MODEL_H5)
    print("    You can use the H5 model directly in spoof_engine.py")
