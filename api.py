"""BASED v2 REST API — predict bankfull channel depth from Q, W, S."""

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(
    title="BASED — Boost-Assisted Stream Estimator for Depth",
    version="2.0.0",
    description=(
        "Predicts bankfull channel depth from discharge, width, and slope "
        "using an XGBoost model trained on 4,315 field-surveyed observations."
    ),
)

model = xgb.Booster()
model.load_model("based_model_v2.ubj")


def _predict(discharge: float, width: float, slope: float) -> float:
    features = pd.DataFrame(
        {
            "log_Q": [np.log10(discharge)],
            "log_w": [np.log10(width)],
            "log_S": [np.log10(slope)],
        },
        dtype=float,
    )
    log_pred = model.predict(xgb.DMatrix(features))
    return float(10 ** log_pred[0])


@app.get("/predict")
def predict(
    discharge: float = Query(..., gt=0, description="Bankfull discharge [m³/s]"),
    width: float = Query(..., gt=0, description="Channel width [m]"),
    slope: float = Query(..., gt=0, le=1, description="Channel slope [m/m]"),
):
    """Return predicted bankfull depth and width/depth ratio."""
    try:
        depth = _predict(discharge, width, slope)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "depth_m": round(depth, 4),
        "width_depth_ratio": round(width / depth, 2),
    }


@app.post("/predict/batch")
def predict_batch(rows: list[dict]):
    """Predict depth for multiple rows. Each dict needs discharge, width, slope."""
    results = []
    for i, row in enumerate(rows):
        try:
            d = row["discharge"]
            w = row["width"]
            s = row["slope"]
            if d <= 0 or w <= 0 or s <= 0:
                raise ValueError("All inputs must be positive")
            depth = _predict(d, w, s)
            results.append({"depth_m": round(depth, 4), "width_depth_ratio": round(w / depth, 2)})
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=422, detail=f"Row {i}: {exc}"
            ) from exc
    return results


@app.get("/health")
def health():
    return {"status": "ok", "model": "based_model_v2.ubj", "version": "2.0.0"}
