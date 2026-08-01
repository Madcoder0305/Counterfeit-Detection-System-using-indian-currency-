# ============================================================
# CNN Model -- Indian Rupee Counterfeit Detection
# Currency: INR (10, 20, 50, 100, 200, 500, 2000)
# Dataset: Kaggle real vs fake Indian currency images
# ============================================================

import os, sys, json, numpy as np
from PIL import Image
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GENUINE_DIR  = os.path.join(PROJECT_ROOT, "data", "images", "genuine")
FAKE_DIR     = os.path.join(PROJECT_ROOT, "data", "images", "fake")
SAVE_DIR     = os.path.join(PROJECT_ROOT, "models", "saved")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "models", "results")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE   = 128
BATCH_SIZE = 32
EPOCHS     = 30


def load_images(directory, label):
    images, labels = [], []
    if not os.path.isdir(directory):
        return np.array([]), np.array([])
    files = sorted([f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    for fname in files:
        try:
            img = Image.open(os.path.join(directory, fname)).convert("L").resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(label)
        except:
            pass
    return np.array(images), np.array(labels)


def prepare_data():
    print("  Loading genuine INR images ...")
    g_imgs, g_labels = load_images(GENUINE_DIR, 0)
    print(f"    -> {len(g_imgs)} genuine")
    print("  Loading fake INR images ...")
    f_imgs, f_labels = load_images(FAKE_DIR, 1)
    print(f"    -> {len(f_imgs)} fake")

    if len(g_imgs) == 0 or len(f_imgs) == 0:
        print("  ERROR: Run scripts/prepare_dataset.py first.")
        sys.exit(1)

    X = np.concatenate([g_imgs, f_imgs]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.concatenate([g_labels, f_labels])

    idx = np.random.RandomState(42).permutation(len(X))
    X, y = X[idx], y[idx]

    n = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    Xtr, ytr = X[:n_train], keras.utils.to_categorical(y[:n_train], 2)
    Xv,  yv  = X[n_train:n_train+n_val], keras.utils.to_categorical(y[n_train:n_train+n_val], 2)
    Xte, yte_raw = X[n_train+n_val:], y[n_train+n_val:]
    yte = keras.utils.to_categorical(yte_raw, 2)

    print(f"  Train: {len(Xtr)} | Val: {len(Xv)} | Test: {len(Xte)}")
    return Xtr, ytr, Xv, yv, Xte, yte, yte_raw


def build_cnn():
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.15),
    ], name="augmentation")

    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        aug,
        # Block 1
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Block 2
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Block 3
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Block 4
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),
        # Classifier
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(2, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train():
    print("\n" + "=" * 60)
    print("  CNN Training -- Indian Rupee (INR) Counterfeit Detection")
    print("  Dataset: Kaggle real vs fake Indian currency notes")
    print("  (includes Feature close-ups + all 7 denominations)")
    print("=" * 60 + "\n")

    Xtr, ytr, Xv, yv, Xte, yte, yte_raw = prepare_data()
    model = build_cnn()
    model.summary()

    # Compute class weights to handle genuine/fake imbalance
    y_train_labels = np.argmax(ytr, axis=1)
    n_genuine = np.sum(y_train_labels == 0)
    n_fake = np.sum(y_train_labels == 1)
    total = n_genuine + n_fake
    class_weight = {
        0: total / (2.0 * n_genuine),
        1: total / (2.0 * n_fake),
    }
    print(f"  Class weights: genuine={class_weight[0]:.3f}, fake={class_weight[1]:.3f}")

    cbs = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    print("\n  Training ...")
    model.fit(
        Xtr, ytr,
        validation_data=(Xv, yv),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        class_weight=class_weight,
        verbose=1,
    )

    # Evaluate
    print("\n  Evaluating on test set ...")
    y_pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
    acc  = accuracy_score(yte_raw, y_pred)
    prec = precision_score(yte_raw, y_pred, zero_division=0)
    rec  = recall_score(yte_raw, y_pred, zero_division=0)
    f1   = f1_score(yte_raw, y_pred, zero_division=0)
    cm   = confusion_matrix(yte_raw, y_pred).tolist()

    print(f"\n  -- CNN Results (INR Kaggle Dataset) --")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion : {cm}")

    # Save
    model_path = os.path.join(SAVE_DIR, "cnn_best.h5")
    model.save(model_path)
    print(f"\n  Model saved -> {model_path}")

    # TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        with open(os.path.join(SAVE_DIR, "cnn_model.tflite"), "wb") as f:
            f.write(converter.convert())
        print("  TFLite exported")
    except Exception as e:
        print(f"  TFLite export failed: {e}")

    # Metrics JSON
    metrics = {
        "model": "CNN",
        "currency": "INR",
        "dataset": "Kaggle preetrank/indian-currency-real-vs-fake-notes-dataset",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
    }
    with open(os.path.join(RESULTS_DIR, "cnn_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("  CNN training complete.\n")


if __name__ == "__main__":
    train()
