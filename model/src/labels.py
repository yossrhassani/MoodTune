"""
Central definitions shared across training, evaluation and the backend.
Keeping this in one file guarantees the label order and model path used
during training match what the backend loads at inference time
(see report, Section 5.2: Model File Traceability).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "saved_models" / "fer.h5"

IMG_SIZE = (100, 100)

# RAF-DB basic emotion partition: numeric folders 1-7 map to these labels,
# in this exact order (report, Table 2.1 / Section 2.1).
CLASS_FOLDERS = ["1", "2", "3", "4", "5", "6", "7"]
CLASS_LABELS = [
    "surprise",
    "fear",
    "disgust",
    "happiness",
    "sadness",
    "anger",
    "neutral",
]

# Expression -> mood group mapping (report, Table 3.3)
EMOTION_TO_MOOD = {
    "surprise": "energetic",
    "fear": "stressed",
    "disgust": "stressed",
    "anger": "stressed",
    "happiness": "positive",
    "sadness": "low",
    "neutral": "calm",
}
MOOD_GROUPS = ["calm", "energetic", "low", "positive", "stressed"]

CONFIDENCE_THRESHOLD = 0.45
