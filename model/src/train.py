"""
Train the MoodTune facial expression CNN (custom architecture, from scratch)
on the RAF-DB basic emotion partition.

This reconstructs the training script behind the reported model
(model/saved_models/fer.h5, 76.60% test accuracy, weighted F1 0.7619),
matching the architecture, hyperparameters and paths described in the
project report (Chapter 3: System Architecture).

Expected data layout:
    model/data/raw/train/{1..7}/*.jpg
    model/data/raw/test/{1..7}/*.jpg
    (1=surprise, 2=fear, 3=disgust, 4=happiness, 5=sadness, 6=anger, 7=neutral)

Run:
    python model/src/train_fer.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Paths and constants (match report: model/data/raw, model/saved_models/fer.h5)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = DATA_DIR / "train"
SAVE_DIR = PROJECT_ROOT / "saved_models"
MODEL_PATH = SAVE_DIR / "fer.h5"

IMG_SIZE = (100, 100)
BATCH_SIZE = 32
NUM_CLASSES = 7
VAL_SPLIT = 0.25
MAX_EPOCHS = 50
SEED = 42

# Folder-to-index mapping used throughout the report (Table 2.1)
CLASSES = ["1", "2", "3", "4", "5", "6", "7"]
CLASS_LABELS = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_data_generators(train_dir: Path):
    """Train/val split from the RAF-DB training partition, with the
    augmentation described in the report (Section 2.3): horizontal flip,
    width/height shift +-10%, rotation +-10 degrees.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=VAL_SPLIT,
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VAL_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        classes=CLASSES,
        subset="training",
        seed=SEED,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        classes=CLASSES,
        subset="validation",
        seed=SEED,
        shuffle=False,
    )
    return train_gen, val_gen


def compute_class_weights(train_dir: Path) -> dict[int, float]:
    """Inverse-frequency class weights (report Section 2.1.1):
    w_c = N / (K * n_c). Gives fear ~6.23x, happiness ~0.37x.
    """
    counts = []
    for cls in CLASSES:
        cls_path = train_dir / cls
        count = len([f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        counts.append(count)
    counts = np.array(counts, dtype=np.float32)
    total = counts.sum()
    weights = total / (NUM_CLASSES * counts)
    class_weights = {i: float(weights[i]) for i in range(NUM_CLASSES)}

    print("\nClass weights (inverse frequency):")
    for i, label in enumerate(CLASS_LABELS):
        print(f"  {label:<10} n={int(counts[i]):>5}  w={class_weights[i]:.3f}")
    print()
    return class_weights


# ---------------------------------------------------------------------------
# Model (report Section 3.2.1 / Figure 3.2 / Table 8.2 — 3,391,431 params)
# ---------------------------------------------------------------------------
def build_model() -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(*IMG_SIZE, 3)),

            layers.Conv2D(64, (3, 3), activation="relu"),   # 1,792 params
            layers.MaxPooling2D((2, 2)),                     # -> 49x49x64

            layers.Conv2D(64, (3, 3), activation="relu"),   # 36,928 params
            layers.MaxPooling2D((2, 2)),                     # -> 23x23x64

            layers.Conv2D(128, (3, 3), activation="relu"),  # 73,856 params
            layers.MaxPooling2D((2, 2)),                     # -> 10x10x128 (per report)

            layers.Flatten(),                                # -> 12,800
            layers.Dense(256, activation="relu"),            # 3,277,056 params
            layers.Dropout(0.3),
            layers.Dense(NUM_CLASSES, activation="softmax"), # 1,799 params
        ],
        name="MoodTune_CNN",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training (report Section 3.3: Adam 0.001, batch 32, 50 epochs max,
# EarlyStopping patience 20, ReduceLROnPlateau, ModelCheckpoint -> fer.h5)
# ---------------------------------------------------------------------------
def get_callbacks(model_path: Path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Expected RAF-DB data at {TRAIN_DIR} (folders 1-7). "
            "Download RAF-DB and place it under model/data/raw/ before training."
        )

    train_gen, val_gen = get_data_generators(TRAIN_DIR)
    class_weights = compute_class_weights(TRAIN_DIR)

    model = build_model()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,} (report: 3,391,431)\n")

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(MODEL_PATH),
    )

    print(f"\nTraining complete. Best model saved to: {MODEL_PATH}")
    print("Run model/src/evaluate.py (or test.py) on model/data/raw/test to reproduce the reported metrics.")


if __name__ == "__main__":
    main()
