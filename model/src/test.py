import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
model = load_model("fer.keras")


# =========================
# LABELS (FIXED ORDER)
# =========================
CATAGORIES = [
    "surprise",
    "fear",
    "disgust",
    "happiness",
    "sadness",
    "anger",
    "neutral"
]


# =========================
# TEST PATH
# =========================
TEST_DIR = r"C:\Users\User\Desktop\PFA\files\data\DATASET\test"


# =========================
# LOAD TEST DATA
# =========================
test_data = []

for folder_name in os.listdir(TEST_DIR):

    if folder_name not in ["1","2","3","4","5","6","7"]:
        continue

    label = int(folder_name) - 1
    folder_path = os.path.join(TEST_DIR, folder_name)

    for img in os.listdir(folder_path):

        img_path = os.path.join(folder_path, img)
        img_arr = cv2.imread(img_path)

        if img_arr is not None:
            img_arr = cv2.resize(img_arr, (100, 100))
            test_data.append([img_arr, label])


# =========================
# SPLIT
# =========================
X_test, Y_test = [], []

for f, l in test_data:
    X_test.append(f)
    Y_test.append(l)

X_test = np.array(X_test) / 255.0
Y_test = np.array(Y_test)


# =========================
# EVALUATE
# =========================
loss, acc = model.evaluate(X_test, Y_test)

print("\n====================")
print("Test Accuracy:", acc)
print("Test Loss:", loss)
print("====================")


# =========================
# SAMPLE PREDICTIONS
# =========================
for i in range(5):

    img = X_test[i]
    true = Y_test[i]

    pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = np.argmax(pred)

    print(f"True: {CATAGORIES[true]} | Pred: {CATAGORIES[pred_class]}")