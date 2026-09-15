"""ReviewIQ — FastAPI backend + static UI, deployed on OpenShift.

Inference is delegated to the RHOAI-served model (OVMS via KServe v2 REST).
"""

from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import analytics, inference

app = FastAPI(title="ReviewIQ", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"
MAX_REVIEWS = 500


class AnalyzeRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=MAX_REVIEWS)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {
        "model": inference.MODEL_ID,
        "model_name": inference.MODEL_NAME,
        "inference": inference.ready(),
        "max_reviews": MAX_REVIEWS,
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    reviews = [r.strip() for r in req.reviews if r.strip()]
    if not reviews:
        raise HTTPException(status_code=422, detail="No non-empty reviews provided.")
    try:
        predictions = inference.predict(reviews)
    except inference.InferenceNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model server returned {e.response.status_code}: {e.response.text[:300]}",
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach model server: {e}")

    return {
        "results": [
            {"review": r, **p} for r, p in zip(reviews, predictions)
        ],
        "summary": analytics.aggregate(reviews, predictions),
    }


app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
