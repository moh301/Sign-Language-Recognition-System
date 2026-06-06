import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

DATA_PATH = "dataset"
MODEL_PATH = "static/model"

os.makedirs(MODEL_PATH, exist_ok=True)

X, y = [], []

# ===== FIX 1: ترتيب ثابت للـ labels =====
labels = sorted(os.listdir(DATA_PATH))

label_map = {label: i for i, label in enumerate(labels)}
reverse_map = {i: label for label, i in label_map.items()}

# ===== LOAD DATA =====
for label in labels:
    folder = os.path.join(DATA_PATH, label)
    for file in os.listdir(folder):
        data = np.load(os.path.join(folder, file))
        X.append(data)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("📊 Data shape:", X.shape)

# ===== FIX 2: Data Augmentation (Noise) =====
noise = np.random.normal(0, 0.01, X.shape)
X_aug = X + noise

X = np.concatenate([X, X_aug])
y = np.concatenate([y, y])

print("📊 After augmentation:", X.shape)

# ===== NORMALIZATION =====
scaler = StandardScaler()
X = scaler.fit_transform(X)

# save scaler
with open(f"{MODEL_PATH}/scaler_v2.pkl", "wb") as f:
    pickle.dump(scaler, f)

# save label map
with open(f"{MODEL_PATH}/label_map_v2.json", "w") as f:
    json.dump(reverse_map, f)

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

# ===== MODEL (Improved) =====
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
    tf.keras.layers.BatchNormalization(),        # 👈 جديد
    tf.keras.layers.Dropout(0.5),                # 👈 أقوى

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ===== EARLY STOPPING (مهم جدًا) =====
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ===== TRAIN =====
history = model.fit(
    X_train, y_train,
    epochs=50,                      # 👈 زودنا epochs
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    batch_size=32
)

# ===== SAVE MODEL =====
model.save(f"{MODEL_PATH}/gesture_model.keras")

print("✅ Model trained & saved!")