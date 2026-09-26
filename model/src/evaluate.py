"""
Evaluate fer.h5 on the RAF-DB test split (model/data/raw/test/{1..7}).

Reproduces the metrics reported in the project report, Section 4.1 /
Table 4.1: test accuracy, test loss, weighted F1, per-class precision/
recall/F1, and the confusion matrix (Figure 4.2).

Run:
    python model/src/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from labels import CLASS_FOLDERS, CLASS_LABELS, IMG_SIZE, MODEL_PATH, PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "raw"
TEST_DIR = DATA_DIR / "test"
SAVE_DIR = PROJECT_ROOT / "saved_models"


def make_test_generator(test_dir: Path):
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Expected RAF-DB test data at {test_dir} (folders 1-7). "
            "Download RAF-DB and place it under model/data/raw/test/."
        )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="sparse",
        classes=CLASS_FOLDERS,
        shuffle=False,
    )


def save_confusion_matrix(y_true, y_pred, class_names, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("MoodTune Facial Expression Confusion Matrix (RAF-DB test)")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    threshold = cm.max() / 2 if cm.max() else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            plt.text(col, row, str(cm[row, col]), ha="center", va="center", color=color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run model/src/train.py first."
        )

    test_gen = make_test_generator(TEST_DIR)
    model = tf.keras.models.load_model(MODEL_PATH)

    loss, accuracy = model.evaluate(test_gen, verbose=0)

    test_gen.reset()
    probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_gen.classes

    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_LABELS, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=CLASS_LABELS, zero_division=0
    )

    confusion_path = SAVE_DIR / "confusion_matrix_fer.png"
    report_path = SAVE_DIR / "evaluation_report_fer.json"
    save_confusion_matrix(y_true, y_pred, CLASS_LABELS, confusion_path)

    report = {
        "model_path": str(MODEL_PATH),
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "test_samples": int(len(y_true)),
        "class_names": CLASS_LABELS,
        "classification_report": report_dict,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nEvaluation complete.")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Test loss:     {loss:.4f}\n")
    print(report_text)
    print(f"Saved confusion matrix: {confusion_path}")
    print(f"Saved report:           {report_path}")


if __name__ == "__main__":
    main()
