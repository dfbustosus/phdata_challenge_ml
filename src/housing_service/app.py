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
from typing import Any, Dict, List, Optional, Union, cast

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sklearn.base import RegressorMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
DEMO_CSV = Path(os.getenv("DEMOGRAPHICS_CSV", "data/zipcode_demographics.csv"))
API_VERSION = "1.0.0"

# Security
security = HTTPBearer(auto_error=False)


class HouseFeatures(BaseModel):
    """House features for prediction (excluding demographic data).
    
    All demographic data is merged internally based on zipcode.
    Input should only contain house-specific features.
    """
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
    
    @field_validator("sqft_above", "sqft_basement")
    @classmethod
    def validate_sqft_consistency(cls, v: int, info) -> int:
        """Validate square footage consistency."""
        if info.data.get("sqft_living") and v > info.data["sqft_living"]:
            raise ValueError("sqft_above/sqft_basement cannot exceed sqft_living")
        return v


class MinimalHouseFeatures(BaseModel):
    """Minimal house features for basic prediction.
    
    This endpoint requires only the most essential features for prediction.
    """
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
    
    records: List[HouseFeatures] = Field(
        ..., min_length=1, max_length=1000, description="List of house records to predict"
    )


class MinimalPredictRequest(BaseModel):
    """Request model for minimal feature batch predictions."""
    model_config = ConfigDict(extra="forbid")
    
    records: List[MinimalHouseFeatures] = Field(
        ..., min_length=1, max_length=1000, description="List of minimal house records to predict"
    )


class PredictionResult(BaseModel):
    """Individual prediction result."""
    prediction: float = Field(..., description="Predicted house price in USD")
    confidence_score: Optional[float] = Field(None, description="Model confidence score")
    zipcode: str = Field(..., description="Zipcode for this prediction")


class PredictResponse(BaseModel):
    """Response model for predictions."""
    model_config = ConfigDict(extra="forbid")
    
    predictions: List[PredictionResult] = Field(..., description="List of predictions")
    model_version: Optional[str] = Field(None, description="Model version used")
    model_type: str = Field(..., description="Type of model used")
    n_records: int = Field(..., description="Number of records processed")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    demographics_loaded: bool = Field(..., description="Whether demographics data is loaded")


class ModelService:
    """Singleton service for model and demographics data management."""
    
    def __init__(self):
        self._model: Optional[RegressorMixin] = None
        self._features: Optional[List[str]] = None
        self._demographics: Optional[pd.DataFrame] = None
        self._model_version: Optional[str] = None
        self._model_type: Optional[str] = None
        
    def load_model_and_features(self, model_dir: Path) -> tuple[RegressorMixin, List[str]]:
        """Load model and features with caching."""
        if self._model is None or self._features is None:
            logger.info(f"Loading model from {model_dir}")
            model_path = model_dir / "model.pkl"
            feats_path = model_dir / "model_features.json"
            
            if not model_path.exists() or not feats_path.exists():
                raise FileNotFoundError(
                    f"Model artifacts not found in {model_dir}. "
                    "Expected 'model.pkl' and 'model_features.json'"
                )
            
            self._model = cast(RegressorMixin, joblib.load(model_path))
            
            with open(feats_path, "r") as f:
                features_obj = json.load(f)
            if not isinstance(features_obj, list) or not all(isinstance(x, str) for x in features_obj):
                raise ValueError("model_features.json must be a JSON list of feature names")
            
            self._features = features_obj
            self._model_version = self._detect_version(model_dir)
            self._model_type = type(self._model).__name__
            
            logger.info(f"Model loaded successfully: {self._model_type}, version: {self._model_version}")
            
        return self._model, self._features
    
    def load_demographics(self, csv_path: Path) -> pd.DataFrame:
        """Load demographics data with caching."""
        if self._demographics is None:
            logger.info(f"Loading demographics from {csv_path}")
            if not csv_path.exists():
                raise FileNotFoundError(f"Demographics CSV not found: {csv_path}")
            
            df = pd.read_csv(csv_path, dtype={"zipcode": str})
            if "zipcode" not in df.columns:
                raise ValueError("Demographics CSV must contain 'zipcode' column")
            
            self._demographics = df.set_index("zipcode")
            logger.info(f"Demographics loaded: {len(self._demographics)} zipcodes")
            
        return self._demographics
    
    def _detect_version(self, model_dir: Path) -> Optional[str]:
        """Detect model version from VERSION file."""
        version_file = model_dir / "VERSION"
        if version_file.exists():
            return version_file.read_text().strip() or None
        return None
    
    @property
    def model_version(self) -> Optional[str]:
        """Get current model version."""
        return self._model_version
    
    @property
    def model_type(self) -> Optional[str]:
        """Get current model type."""
        return self._model_type
    
    @property
    def is_loaded(self) -> bool:
        """Check if model and demographics are loaded."""
        return (
            self._model is not None 
            and self._features is not None 
            and self._demographics is not None
        )


# Global model service instance
model_service = ModelService()


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    """Dependency injection for model service."""
    return model_service





def _prepare_features(
    records: List[Union[HouseFeatures, MinimalHouseFeatures]], 
    demo_df: pd.DataFrame, 
    features: List[str],
    is_minimal: bool = False
) -> pd.DataFrame:
    """Prepare features for prediction by merging house data with demographics.
    
    Args:
        records: List of house feature records
        demo_df: Demographics DataFrame indexed by zipcode
        features: Required model features
        is_minimal: Whether this is for minimal features endpoint
        
    Returns:
        DataFrame with all required features for prediction
        
    Raises:
        ValueError: If required features are missing or data is invalid
    """
    # Convert Pydantic models to dictionaries
    rows = [record.model_dump() for record in records]
    base_df = pd.DataFrame(rows)
    
    # Validate zipcode column exists
    if "zipcode" not in base_df.columns:
        raise ValueError("Input rows must include 'zipcode'")
    
    # For minimal features, add default values for missing house features
    if is_minimal:
        # Add default values for missing house features required by the model
        house_features = ["sqft_lot", "floors", "sqft_above", "sqft_basement"]
        for feature in house_features:
            if feature not in base_df.columns:
                if feature == "sqft_lot":
                    base_df[feature] = base_df["sqft_living"] * 2  # Reasonable default
                elif feature == "floors":
                    base_df[feature] = 1.0  # Default to single story
                elif feature == "sqft_above":
                    base_df[feature] = base_df["sqft_living"]  # Assume no basement
                elif feature == "sqft_basement":
                    base_df[feature] = 0  # Assume no basement
    
    # Merge with demographics data
    merged = base_df.merge(
        demo_df.reset_index(), 
        how="left", 
        on="zipcode", 
        validate="m:1"
    )
    
    # Handle missing demographics with median imputation instead of zeros
    demo_columns = demo_df.columns.tolist()
    for col in demo_columns:
        if col in merged.columns:
            median_val = demo_df[col].median()
            merged[col] = merged[col].fillna(median_val)
            logger.debug(f"Filled {merged[col].isna().sum()} missing values in {col} with median {median_val}")
    
    # Ensure all required features exist
    missing = [c for c in features if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing required model features: {', '.join(missing)}")
    
    # Select only the required features in the correct order
    X = merged[features]
    
    # Final validation - check for any remaining NaN values
    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN values found in features: {', '.join(nan_cols)}")
    
    return X





# FastAPI app initialization with proper configuration
app = FastAPI(
    title="Housing Price Prediction API",
    description="RESTful API for predicting house prices using machine learning",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize model and demographics data on startup."""
    try:
        service = get_model_service()
        service.load_model_and_features(MODEL_DIR)
        service.load_demographics(DEMO_CSV)
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
def health(service: ModelService = Depends(get_model_service)) -> HealthResponse:
    """Health check endpoint.
    
    Returns:
        HealthResponse: Current service status and model information
    """
    return HealthResponse(
        status="healthy" if service.is_loaded else "unhealthy",
        version=API_VERSION,
        model_loaded=service._model is not None,
        demographics_loaded=service._demographics is not None
    )


@app.post("/v1/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    service: ModelService = Depends(get_model_service)
) -> PredictResponse:
    """Main prediction endpoint for house price prediction.
    
    Accepts house features (excluding demographic data) and returns predictions.
    Demographic data is merged internally based on zipcode.
    
    Args:
        req: Request containing list of house feature records
        service: Injected model service
        
    Returns:
        PredictResponse: Predictions with metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    import time
    start_time = time.time()
    
    try:
        # Load model and demographics
        model, features = service.load_model_and_features(MODEL_DIR)
        demo_df = service.load_demographics(DEMO_CSV)
        
        # Prepare features and make predictions
        X = _prepare_features(req.records, demo_df, features, is_minimal=False)
        predictions = model.predict(X)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Format response
        prediction_results = [
            PredictionResult(
                prediction=float(pred),
                confidence_score=None,  # KNN doesn't provide confidence scores
                zipcode=record.zipcode
            )
            for pred, record in zip(predictions, req.records)
        ]
        
        logger.info(f"Processed {len(req.records)} predictions in {processing_time:.2f}ms")
        
        return PredictResponse(
            predictions=prediction_results,
            model_version=service.model_version,
            model_type=service.model_type or "Unknown",
            n_records=len(req.records),
            processing_time_ms=processing_time
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model artifacts not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service unavailable: {str(e)}"
        )
    except ValueError as e:
        logger.warning(f"Invalid input data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )


@app.post("/v1/predict-minimal", response_model=PredictResponse)
def predict_minimal(
    req: MinimalPredictRequest,
    service: ModelService = Depends(get_model_service)
) -> PredictResponse:
    """Bonus endpoint for minimal feature prediction.
    
    Accepts only essential house features and provides reasonable defaults
    for missing features required by the model.
    
    Args:
        req: Request containing list of minimal house feature records
        service: Injected model service
        
    Returns:
        PredictResponse: Predictions with metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    import time
    start_time = time.time()
    
    try:
        # Load model and demographics
        model, features = service.load_model_and_features(MODEL_DIR)
        demo_df = service.load_demographics(DEMO_CSV)
        
        # Prepare features with defaults for missing house features
        X = _prepare_features(req.records, demo_df, features, is_minimal=True)
        predictions = model.predict(X)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Format response
        prediction_results = [
            PredictionResult(
                prediction=float(pred),
                confidence_score=None,  # KNN doesn't provide confidence scores
                zipcode=record.zipcode
            )
            for pred, record in zip(predictions, req.records)
        ]
        
        logger.info(f"Processed {len(req.records)} minimal predictions in {processing_time:.2f}ms")
        
        return PredictResponse(
            predictions=prediction_results,
            model_version=service.model_version,
            model_type=service.model_type or "Unknown",
            n_records=len(req.records),
            processing_time_ms=processing_time
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model artifacts not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service unavailable: {str(e)}"
        )
    except ValueError as e:
        logger.warning(f"Invalid input data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )


@app.get("/v1/model-info")
def model_info(service: ModelService = Depends(get_model_service)) -> Dict[str, Any]:
    """Get information about the loaded model.
    
    Returns:
        Dict containing model metadata and feature information
    """
    try:
        model, features = service.load_model_and_features(MODEL_DIR)
        demo_df = service.load_demographics(DEMO_CSV)
        
        return {
            "model_type": service.model_type,
            "model_version": service.model_version,
            "n_features": len(features),
            "features": features,
            "house_features": ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_above", "sqft_basement"],
            "demographic_features": [f for f in features if f not in ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_above", "sqft_basement"]],
            "n_zipcodes": len(demo_df),
            "api_version": API_VERSION
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model information unavailable"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
