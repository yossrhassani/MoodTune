"""
MoodTune backend API.

Routes match the report (Table 3.2):
    GET  /         redirects to the browser app
    GET  /app       serves the standalone browser client
    GET  /health    API status
    POST /analyze   main inference endpoint
    POST /predict   alias for /analyze

Response schema for /analyze matches report Table 5.2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from . import coaching, music, predict

app = FastAPI(title="MoodTune API", version="1.0.0")

WEB_APP_PATH = Path(__file__).resolve().parent / "web_app.html"


class AnalyzeResponse(BaseModel):
    emotion: str
    mood: str
    confidence: float
    is_uncertain: bool
    face_detected: bool
    all_scores: Dict[str, float]
    mood_scores: Dict[str, float]
    guidance: dict
    music: dict


@app.get("/")
def root():
    return RedirectResponse(url="/app")


@app.get("/app")
def serve_app():
    if not WEB_APP_PATH.exists():
        raise HTTPException(status_code=404, detail="web_app.html not found")
    return FileResponse(WEB_APP_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "service": "moodtune-backend"}


async def _run_analysis(file: UploadFile) -> AnalyzeResponse:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    try:
        result = predict.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    guidance = coaching.get_coaching(result["emotion"])
    recommendations = music.get_recommendations(result["mood"])

    return AnalyzeResponse(
        **result,
        guidance=guidance,
        music=recommendations,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    return await _run_analysis(file)


@app.post("/predict", response_model=AnalyzeResponse)
async def predict_alias(file: UploadFile = File(...)):
    return await _run_analysis(file)
