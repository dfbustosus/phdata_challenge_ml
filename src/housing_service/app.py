from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from sklearn.base import RegressorMixin

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
DEMO_CSV = Path(os.getenv("DEMOGRAPHICS_CSV", "data/zipcode_demographics.csv"))


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]

    @field_validator("records")
    @classmethod
    def validate_records(cls, v: List[Dict[str, Any]]):
        if not v:
            raise ValueError("records must be a non-empty list")
        for i, rec in enumerate(v):
            if "zipcode" not in rec:
                raise ValueError(f"record {i} missing required field 'zipcode'")
        return v


class PredictResponse(BaseModel):
    predictions: List[float]
    model_version: Optional[str] = None
    n_records: int


def _load_model_and_features(model_dir: Path) -> tuple[RegressorMixin, list[str]]:
    model_path = model_dir / "model.pkl"
    feats_path = model_dir / "model_features.json"
    if not model_path.exists() or not feats_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {model_dir}. "
            "Expected 'model.pkl' and 'model_features.json'"
        )
    model = cast(RegressorMixin, joblib.load(model_path))
    with open(feats_path, "r") as f:
        features_obj = json.load(f)
    if not isinstance(features_obj, list) or not all(isinstance(x, str) for x in features_obj):
        raise ValueError("model_features.json must be a JSON list of feature names")
    features: list[str] = features_obj
    return model, features


def _load_demographics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Demographics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "zipcode" not in df.columns:
        raise ValueError("Demographics CSV must contain 'zipcode' column")
    return df.set_index("zipcode")


def _prepare_features(
    rows: List[Dict[str, Any]], demo_df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    base_df = pd.DataFrame(rows)
    if "zipcode" not in base_df.columns:
        raise ValueError("Input rows must include 'zipcode'")
    merged = base_df.merge(demo_df.reset_index(), how="left", on="zipcode", validate="m:1")
    # Missing demographics -> fillna 0 (template behavior, can be improved later)
    merged = merged.fillna(0)
    # Ensure all required features exist
    missing = [c for c in features if c not in merged.columns]
    if missing:
        raise ValueError("Missing required model features in joined data: " + ", ".join(missing))
    X = merged[features]
    return X


def _detect_version(model_dir: Path) -> Optional[str]:
    version_file = model_dir / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip() or None
    return None


app = FastAPI(title="Housing Price Service", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        model, features = _load_model_and_features(MODEL_DIR)
        demo = _load_demographics(DEMO_CSV)
        X = _prepare_features(req.records, demo, features)
        preds = model.predict(X)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))

    version = _detect_version(MODEL_DIR)
    return PredictResponse(
        predictions=[float(p) for p in preds], model_version=version, n_records=len(req.records)
    )
