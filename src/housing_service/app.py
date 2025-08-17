"""Housing Price Prediction API.

This module provides RESTful endpoints for housing price prediction using a trained
machine learning model. The API accepts house features and merges demographic data
internally based on zipcode.

Author: ML Engineering Team
Version: 1.0.0
Date: 2025-08-17
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .logger import setup_logger

# Configuration
MODEL_DIR = os.getenv("MODEL_DIR", "model")
DEMOGRAPHICS_CSV = os.getenv("DEMOGRAPHICS_CSV", "data/zipcode_demographics.csv")
DEFAULT_MODEL_VERSION = os.getenv("DEFAULT_MODEL_VERSION", "v2")

# Initialize logger
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Housing Price Prediction API",
    description="RESTful API for predicting housing prices with demographic data integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class HouseFeatures(BaseModel):
    """House features for prediction (excluding demographic data)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    bedrooms: int = Field(..., ge=0, le=50, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=20, description="Number of bathrooms")
    sqft_living: int = Field(..., ge=100, le=50000, description="Square feet of living space")
    sqft_lot: int = Field(..., ge=100, le=1000000, description="Square feet of lot")
    floors: float = Field(..., ge=1, le=10, description="Number of floors")
    sqft_above: int = Field(..., ge=0, le=50000, description="Square feet above ground")
    sqft_basement: int = Field(..., ge=0, le=50000, description="Square feet of basement")
    zipcode: str = Field(..., min_length=5, max_length=5, description="5-digit zipcode")

    @field_validator("zipcode")
    @classmethod
    def validate_zipcode(cls, v: str) -> str:
        """Validate zipcode format."""
        if not v.isdigit():
            raise ValueError("zipcode must contain only digits")
        return v


class MinimalHouseFeatures(BaseModel):
    """Minimal house features for basic prediction."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    bedrooms: int = Field(..., ge=0, le=50, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=20, description="Number of bathrooms")
    sqft_living: int = Field(..., ge=100, le=50000, description="Square feet of living space")
    zipcode: str = Field(..., min_length=5, max_length=5, description="5-digit zipcode")

    @field_validator("zipcode")
    @classmethod
    def validate_zipcode(cls, v: str) -> str:
        """Validate zipcode format."""
        if not v.isdigit():
            raise ValueError("zipcode must contain only digits")
        return v


class PredictRequest(BaseModel):
    """Request model for batch predictions."""

    model_config = ConfigDict(extra="forbid")

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=1000, description="List of house records to predict"
    )


class MinimalPredictRequest(BaseModel):
    """Request model for minimal feature batch predictions."""

    model_config = ConfigDict(extra="forbid")

    records: List[MinimalHouseFeatures] = Field(
        ..., min_length=1, max_length=1000, description="List of minimal house records to predict"
    )


class PredictResponse(BaseModel):
    """Response model for predictions."""

    model_config = ConfigDict(extra="forbid")

    predictions: List[float] = Field(..., description="List of predicted prices")
    model_version: str = Field(..., description="Model version used")
    model_type: str = Field(..., description="Type of model used")
    n_records: int = Field(..., description="Number of records processed")
    feature_count: int = Field(..., description="Number of features used")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    demographics_loaded: bool = Field(..., description="Whether demographics data is loaded")


class ModelService:
    """Singleton service for model and demographics loading with multi-version support."""

    _instance = None
    _models = {}  # Cache for multiple model versions
    _features = {}  # Cache for multiple feature sets
    _demographics = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_dir: str, version: str = "v2") -> tuple[Any, List[str]]:
        """Load model and features with caching and version support."""
        cache_key = f"{model_dir}_{version}"

        if cache_key not in self._models or cache_key not in self._features:
            version_dir = Path(model_dir) / version
            model_path = version_dir / "model.pkl"
            features_path = version_dir / "model_features.json"

            if not model_path.exists():
                raise FileNotFoundError(f"Model v{version} not found at {model_path}")
            if not features_path.exists():
                raise FileNotFoundError(f"Features v{version} not found at {features_path}")

            logger.info(f"Loading model version {version} from {version_dir}")
            self._models[cache_key] = joblib.load(model_path)

            with open(features_path, "r") as f:
                self._features[cache_key] = json.load(f)

        return self._models[cache_key], self._features[cache_key]

    def load_demographics(self, demographics_path: str) -> pd.DataFrame:
        """Load demographics data with caching."""
        if self._demographics is None:
            if not Path(demographics_path).exists():
                raise FileNotFoundError(f"Demographics not found at {demographics_path}")

            logger.info(f"Loading demographics from {demographics_path}")
            self._demographics = pd.read_csv(demographics_path, dtype={"zipcode": str})

        return self._demographics

    def get_available_versions(self, model_dir: str) -> List[str]:
        """Get list of available model versions."""
        model_path = Path(model_dir)
        if not model_path.exists():
            return []

        versions = []
        for item in model_path.iterdir():
            if item.is_dir() and (item / "model.pkl").exists():
                versions.append(item.name)

        return sorted(versions)

    def get_model_metadata(self, model_dir: str, version: str = "v2") -> Dict[str, Any]:
        """Get model metadata from versioned directory."""
        try:
            version_dir = Path(model_dir) / version
            metadata_file = version_dir / "model_metadata.json"

            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                return metadata

            return {"version": version, "model_type": "Unknown"}
        except Exception:
            return {"version": version, "model_type": "Unknown"}

    @property
    def is_loaded(self) -> bool:
        """Check if model and demographics are loaded."""
        return self._models and self._features and self._demographics is not None


# Global model service instance
model_service = ModelService()


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    """Dependency injection for model service."""
    return model_service


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as the training pipeline."""
    import numpy as np

    # Create a copy to avoid modifying original
    df_eng = df.copy()

    # Feature ratios
    df_eng["sqft_living_to_lot_ratio"] = df_eng["sqft_living"] / (df_eng["sqft_lot"] + 1)
    df_eng["bathroom_to_bedroom_ratio"] = df_eng["bathrooms"] / (df_eng["bedrooms"] + 1)
    df_eng["above_to_living_ratio"] = df_eng["sqft_above"] / (df_eng["sqft_living"] + 1)
    df_eng["basement_to_living_ratio"] = df_eng["sqft_basement"] / (df_eng["sqft_living"] + 1)

    # Room density
    df_eng["room_density"] = (df_eng["bedrooms"] + df_eng["bathrooms"]) / (
        df_eng["sqft_living"] + 1
    )

    # Size categories (this creates categorical data that needs dummy encoding)
    df_eng["size_category"] = pd.cut(
        df_eng["sqft_living"],
        bins=[0, 1000, 1500, 2000, 3000, float("inf")],
        labels=[1, 2, 3, 4, 5],
    )

    # Log transforms for skewed features (EXCLUDE TARGET VARIABLE)
    skewed_features = ["sqft_living", "sqft_lot"]
    for feature in skewed_features:
        if feature in df_eng.columns:
            df_eng[f"{feature}_log"] = np.log1p(df_eng[feature])

    # Handle categorical variables the same way as training
    categorical_cols = df_eng.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        logger.info(f"Found categorical columns before dummy encoding: {list(categorical_cols)}")
        df_eng = pd.get_dummies(df_eng, columns=categorical_cols, drop_first=True)
        logger.info(f"After dummy encoding: {list(df_eng.columns)}")

    return df_eng


def prepare_features(
    records: List[Dict[str, Any]], demo_df: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    """Prepare features for prediction by merging house data with demographics.

    Args:
        records: List of house feature records
        demo_df: Demographics DataFrame indexed by zipcode
        features: Required model features

    Returns:
        DataFrame with all required features for prediction

    Raises:
        ValueError: If required features are missing or data is invalid
    """
    # Convert Pydantic models to dictionaries
    rows = records
    base_df = pd.DataFrame(rows)

    # Validate zipcode column exists
    if "zipcode" not in base_df.columns:
        raise ValueError("Input rows must include 'zipcode'")

    # Ensure zipcode is string type for proper merging
    base_df["zipcode"] = base_df["zipcode"].astype(str)

    # Merge with demographics data
    merged = base_df.merge(demo_df, how="left", on="zipcode", validate="m:1")

    # Handle missing demographics with median imputation instead of zeros
    # Exclude zipcode from median imputation as it's categorical
    demo_columns = [col for col in demo_df.columns.tolist() if col != "zipcode"]
    for col in demo_columns:
        if col in merged.columns and merged[col].dtype in ["float64", "int64"]:
            median_val = demo_df[col].median()
            merged[col] = merged[col].fillna(median_val)
            logger.debug(
                f"Filled {merged[col].isna().sum()} missing values in {col} with median {median_val}"
            )

    # Apply the same feature engineering as training pipeline
    merged = apply_feature_engineering(merged)

    # Drop zipcode column as it's only used for merging, not prediction
    if "zipcode" in merged.columns:
        merged = merged.drop(columns=["zipcode"])

    # Debug: Log what we have after feature engineering
    logger.info(f"After feature engineering - columns: {list(merged.columns)}")
    logger.info(f"Required model features: {features}")

    # Ensure all required features exist (excluding zipcode)
    model_features = [f for f in features if f != "zipcode"]
    missing = [c for c in model_features if c not in merged.columns]
    if missing:
        logger.error(f"Missing required model features: {missing}")
        logger.error(f"Available columns: {list(merged.columns)}")
        raise ValueError(f"Missing required model features: {', '.join(missing)}")

    # Select only the required features in the correct order
    X = merged[model_features].copy()

    # Debug: Log final feature matrix
    logger.info(f"Final feature matrix shape: {X.shape}")
    logger.info(f"Final feature columns: {list(X.columns)}")
    logger.info(f"Data types: {X.dtypes.to_dict()}")

    # Ensure all columns are numeric
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            logger.error(f"Found non-numeric column {col} with values: {X[col].unique()}")
            raise ValueError(f"Non-numeric column found: {col}")

    # Final validation - check for any remaining NaN values
    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        logger.error(f"NaN values found in features: {nan_cols}")
        raise ValueError(f"NaN values found in features: {', '.join(nan_cols)}")

    return X


# FastAPI app initialization with proper configuration


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        logger.info("Starting Housing Price Prediction API...")

        # Pre-load default model and demographics for faster first requests
        model_service.load_model(MODEL_DIR, DEFAULT_MODEL_VERSION)
        model_service.load_demographics(DEMOGRAPHICS_CSV)

        available_versions = model_service.get_available_versions(MODEL_DIR)
        logger.info(
            f"✅ Model and demographics loaded successfully. Available versions: {available_versions}"
        )
    except Exception as e:
        logger.warning(f"⚠️  Warning: Could not pre-load model: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    try:
        # Try to load model and demographics to verify service health
        model_service.load_model(MODEL_DIR, DEFAULT_MODEL_VERSION)
        model_service.load_demographics(DEMOGRAPHICS_CSV)
        is_healthy = model_service.is_loaded
    except Exception:
        is_healthy = False

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        version="2.0.0",
        model_loaded=model_service.is_loaded,
        demographics_loaded=model_service._demographics is not None,
    )


@app.post("/v1/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    model_version: str = Query(
        default=DEFAULT_MODEL_VERSION, description="Model version to use (v1, v2, etc.)"
    ),
) -> PredictResponse:
    """Main prediction endpoint with demographic data integration and version support."""
    try:
        logger.info(
            f"Processing prediction request with {len(request.records)} records using model {model_version}"
        )

        # Load model and demographics
        try:
            logger.info("Step 1: Loading model and demographics...")
            model, features = model_service.load_model(MODEL_DIR, model_version)
            demographics = model_service.load_demographics(DEMOGRAPHICS_CSV)
            logger.info(f"Step 1 SUCCESS: Loaded model with {len(features)} features")
        except Exception as e:
            logger.error(f"Step 1 FAILED: {e}")
            raise

        # Prepare features
        try:
            logger.info("Step 2: Preparing features...")
            logger.info(f"Input records: {request.records}")
            X = prepare_features(request.records, demographics, features)
            logger.info(f"Step 2 SUCCESS: Prepared features with shape {X.shape}")
        except Exception as e:
            logger.error(f"Step 2 FAILED: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

        # Debug: Log exactly what we're passing to the model
        logger.info(f"Step 3: About to call model.predict() with:")
        logger.info(f"  X.shape: {X.shape}")
        logger.info(f"  X.columns: {list(X.columns)}")
        logger.info(f"  X.dtypes: {X.dtypes.to_dict()}")
        logger.info(f"  Expected features: {features}")
        logger.info(f"  Sample X values:\n{X.head()}")

        # Make predictions
        try:
            logger.info("Step 3: Making predictions...")
            predictions = model.predict(X)
            logger.info(f"Step 3 SUCCESS: Generated {len(predictions)} predictions")
        except Exception as e:
            logger.error(f"Step 3 FAILED: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

        # Get model metadata
        metadata = model_service.get_model_metadata(MODEL_DIR, model_version)

        return PredictResponse(
            predictions=[float(pred) for pred in predictions],
            model_version=metadata.get("version", model_version),
            model_type=metadata.get("model_type", "Unknown"),
            n_records=len(request.records),
            feature_count=len(features),
        )

    except FileNotFoundError as e:
        logger.error(f"Model service unavailable: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service unavailable: {str(e)}",
        )
    except ValueError as e:
        logger.error(f"Invalid input data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}"
        )


@app.post("/v1/predict-minimal", response_model=PredictResponse)
async def predict_minimal(
    request: MinimalPredictRequest,
    model_version: str = Query(
        default=DEFAULT_MODEL_VERSION, description="Model version to use (v1, v2, etc.)"
    ),
) -> PredictResponse:
    """Minimal features prediction endpoint with intelligent defaults and version support."""
    try:
        logger.info(
            f"Processing minimal prediction request with {len(request.records)} records using model {model_version}"
        )

        # Load model and demographics
        model, features = model_service.load_model(MODEL_DIR, model_version)
        demographics = model_service.load_demographics(DEMOGRAPHICS_CSV)

        # Convert minimal request to full format with defaults
        full_records = []
        for record in request.records:
            full_record = {
                "bedrooms": record.bedrooms,
                "bathrooms": record.bathrooms,
                "sqft_living": record.sqft_living,
                "zipcode": record.zipcode,
                # Intelligent defaults based on typical values
                "sqft_lot": 7500,  # Median lot size
                "floors": 1.0,
                "sqft_above": record.sqft_living,  # Assume no basement
                "sqft_basement": 0,
            }
            full_records.append(full_record)

        # Prepare features
        X = prepare_features(full_records, demographics, features)

        # Make predictions
        predictions = model.predict(X)
        logger.info(f"Generated {len(predictions)} minimal predictions")

        # Get model metadata
        metadata = model_service.get_model_metadata(MODEL_DIR, model_version)

        return PredictResponse(
            predictions=[float(pred) for pred in predictions],
            model_version=metadata.get("version", model_version),
            model_type=metadata.get("model_type", "Unknown"),
            n_records=len(request.records),
            feature_count=len(features),
        )

    except FileNotFoundError as e:
        logger.error(f"Model service unavailable: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service unavailable: {str(e)}",
        )
    except ValueError as e:
        logger.error(f"Invalid input data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}"
        )


@app.get("/v1/model-info")
async def get_model_info(
    model_version: str = Query(
        default=DEFAULT_MODEL_VERSION, description="Model version to get info for"
    )
) -> Dict[str, Any]:
    """Get information about the specified model version."""
    try:
        metadata = model_service.get_model_metadata(MODEL_DIR, model_version)
        _, features = model_service.load_model(MODEL_DIR, model_version)
        available_versions = model_service.get_available_versions(MODEL_DIR)

        return {
            "model_version": metadata.get("version", model_version),
            "model_type": metadata.get("model_type", "Unknown"),
            "feature_count": len(features),
            "features": features[:10],  # Show first 10 features
            "available_versions": available_versions,
            "default_version": DEFAULT_MODEL_VERSION,
            "api_version": "2.0.0",
            "endpoints": ["/v1/predict", "/v1/predict-minimal", "/v1/model-info", "/health"],
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
