"""
Inference pipeline used by the FastAPI backend: face detection, preprocessing
that matches training exactly, prediction, and mood aggregation.

Implements the face-crop algorithm from the report (Algorithm 1, Section 5.5)
and the confidence-threshold behaviour from Section 3.5.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import tensorflow as tf

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "model" / "src"))
from labels import CLASS_LABELS, CONFIDENCE_THRESHOLD, EMOTION_TO_MOOD, IMG_SIZE, MODEL_PATH, MOOD_GROUPS  # noqa: E402

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
FACE_PADDING_RATIO = 0.18


@lru_cache(maxsize=1)
def get_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(str(MODEL_PATH))


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR image, matching OpenCV's default
    channel order (the model was trained on BGR arrays, report Section 2.2)."""
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes.")
    return image


def detect_and_crop_face(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Algorithm 1 (report, Section 5.5): detect the largest face, expand
    the box by 18% padding, fall back to the full image if none is found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        return image, False

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad = int(FACE_PADDING_RATIO * max(w, h))

    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, image.shape[1])
    y1 = min(y + h + pad, image.shape[0])

    return image[y0:y1, x0:x1], True


def preprocess(face_bgr: np.ndarray) -> np.ndarray:
    """Resize to the model's input resolution and normalise to [0, 1],
    matching training preprocessing exactly (report, Figure 2.1)."""
    resized = cv2.resize(face_bgr, IMG_SIZE)
    normalised = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalised, axis=0)


def compute_mood_scores(class_probs: dict[str, float]) -> dict[str, float]:
    """Sum expression probabilities per mood group (report, Table 3.3 and
    Section 5.6) rather than relying on a single top class."""
    mood_scores = {mood: 0.0 for mood in MOOD_GROUPS}
    for emotion, prob in class_probs.items():
        mood_scores[EMOTION_TO_MOOD[emotion]] += prob
    return mood_scores


def predict(image_bytes: bytes) -> dict:
    image = decode_image(image_bytes)
    face, face_detected = detect_and_crop_face(image)
    tensor = preprocess(face)

    model = get_model()
    probs = model.predict(tensor, verbose=0)[0]

    class_probs = {label: float(p) for label, p in zip(CLASS_LABELS, probs)}
    top_index = int(np.argmax(probs))
    emotion = CLASS_LABELS[top_index]
    confidence = float(probs[top_index])

    mood_scores = compute_mood_scores(class_probs)
    mood = max(mood_scores, key=mood_scores.get)

    return {
        "emotion": emotion,
        "mood": mood,
        "confidence": confidence,
        "is_uncertain": confidence < CONFIDENCE_THRESHOLD,
        "face_detected": face_detected,
        "all_scores": class_probs,
        "mood_scores": mood_scores,
    }
