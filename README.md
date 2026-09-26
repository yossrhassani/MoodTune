# MoodTune — Facial Expression Recognition & Music Recommendation

Final year project (PFA), INSAT, Academic Year 2025/2026.

MoodTune captures a facial expression, classifies it with a custom CNN trained on
RAF-DB, aggregates the result into one of five mood groups, and recommends music
via the Deezer API. Available as a browser app and an Expo React Native mobile app.

**Reported results (RAF-DB test set, 3,068 images):** 76.60% accuracy, weighted F1 0.7619.

## Repository organization

```
model/
  src/
    labels.py      # shared class labels, mood mapping, model path
    train.py        # trains the CNN from scratch, saves saved_models/fer.h5
    evaluate.py      # reproduces the reported metrics + confusion matrix
    test.py          # quick sanity check on a handful of test predictions
backend/
  main.py            # FastAPI app: /health, /app, /analyze, /predict
  predict.py          # face detection, preprocessing, inference, mood aggregation
  music.py             # Deezer search, dedup, fallback
  coaching.py           # short supportive text per expression
  web_app.html            # standalone browser client
mobile/
  App.js               # Expo app: camera capture, result tabs, music playback
  api.js                 # backend URL config + API calls
docs/
  MoodTune_Presentation.pdf
requirements.txt
```

## A note on how this repository was assembled

Some of this code is original project work; some was reconstructed afterward from
the project report to fill gaps in what was available at publishing time. Being upfront
about which is which:

- **Faithful to the original project:** `model/src/test.py`, the overall architecture,
  hyperparameters, API schema, mood-aggregation logic, and UI flow all come directly
  from the project report and presentation.
- **Reconstructed to match the report's exact specification:** `model/src/train.py` was
  rebuilt to match the report's documented architecture (verified to produce exactly
  3,391,431 parameters, matching the report's parameter table) after the original
  training script could not be located. `model/src/evaluate.py`, the full `backend/`
  directory, and `mobile/App.js` were similarly rebuilt from the report's architecture
  diagrams, endpoint tables, and JSON schema, since the original files were not
  available when this repository was published.
- Running `train.py` fresh on RAF-DB will reproduce the architecture exactly but is
  not guaranteed to reproduce the exact 76.60%/F1 0.7619 figures bit-for-bit, since
  those depend on the original random split and initialization.

## Setup

```bash
pip install -r requirements.txt
```

Download RAF-DB and place it under:
```
model/data/raw/train/{1..7}/*.jpg
model/data/raw/test/{1..7}/*.jpg
```
(1=surprise, 2=fear, 3=disgust, 4=happiness, 5=sadness, 6=anger, 7=neutral)

## Train and evaluate

```bash
python model/src/train.py
python model/src/evaluate.py
```

## Run the backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Then open `http://localhost:8000/app` for the browser client.

## Run the mobile app

```bash
cd mobile
npx expo start
```
Update `BASE_URL` in `mobile/api.js` to your machine's LAN IP so a physical phone
can reach the backend.

## Ethical note

MoodTune predicts a visible facial expression, not a diagnosis of internal emotional
state. Results are shown with a confidence score and are explicitly framed as
supportive, not clinical.
