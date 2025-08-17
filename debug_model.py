#!/usr/bin/env python3
"""
Debug script to examine the model and understand the categorical conversion error.
"""

import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path


def debug_model():
    """Debug the model to understand the categorical conversion error."""

    # Load model and features
    model_path = Path("model/v2/model.pkl")
    features_path = Path("model/v2/model_features.json")
    demographics_path = Path("data/zipcode_demographics.csv")

    print("🔍 Loading model and features...")
    model = joblib.load(model_path)
    with open(features_path, "r") as f:
        features = json.load(f)

    print(f"Model type: {type(model)}")
    print(f"Model steps (if pipeline): {getattr(model, 'steps', 'Not a pipeline')}")
    print(f"Expected features ({len(features)}): {features}")

    # Load demographics
    demo_df = pd.read_csv(demographics_path)
    demo_df["zipcode"] = demo_df["zipcode"].astype(str)
    print(f"Demographics shape: {demo_df.shape}")
    print(f"Demographics columns: {list(demo_df.columns)}")

    # Create test input similar to API
    test_input = {
        "bedrooms": 3,
        "bathrooms": 2.5,
        "sqft_living": 2000,
        "sqft_lot": 8000,
        "floors": 2.0,
        "sqft_above": 2000,
        "sqft_basement": 0,
        "zipcode": "98001",
    }

    print(f"\n🧪 Testing with input: {test_input}")

    # Convert to DataFrame
    base_df = pd.DataFrame([test_input])
    base_df["zipcode"] = base_df["zipcode"].astype(str)
    print(f"Base DataFrame:\n{base_df}")
    print(f"Base DataFrame dtypes:\n{base_df.dtypes}")

    # Merge with demographics
    print("\n🔗 Merging with demographics...")
    merged = base_df.merge(demo_df, how="left", on="zipcode", validate="m:1")
    print(f"Merged shape: {merged.shape}")
    print(f"Merged columns: {list(merged.columns)}")

    # Apply feature engineering
    print("\n⚙️ Applying feature engineering...")
    df_eng = apply_feature_engineering(merged)
    print(f"After feature engineering shape: {df_eng.shape}")
    print(f"After feature engineering columns: {list(df_eng.columns)}")
    print(f"Data types:\n{df_eng.dtypes}")

    # Drop zipcode
    if "zipcode" in df_eng.columns:
        df_eng = df_eng.drop(columns=["zipcode"])
        print(f"After dropping zipcode: {list(df_eng.columns)}")

    # Check for required features
    model_features = [f for f in features if f != "zipcode"]
    missing = [c for c in model_features if c not in df_eng.columns]
    if missing:
        print(f"❌ Missing features: {missing}")
        return

    # Select model features
    X = df_eng[model_features]
    print("\n📊 Final feature matrix:")
    print(f"Shape: {X.shape}")
    print(f"Columns: {list(X.columns)}")
    print(f"Data types:\n{X.dtypes}")
    print(f"Sample values:\n{X.head()}")

    # Check for categorical columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        print(f"❌ Found categorical columns: {list(categorical_cols)}")
        for col in categorical_cols:
            print(f"   {col}: {X[col].unique()}")
        return

    # Try prediction
    print("\n🎯 Attempting prediction...")
    try:
        prediction = model.predict(X)
        print(f"✅ Prediction successful: {prediction}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        traceback.print_exc()


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as the training pipeline."""
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
        print(f"Found categorical columns before dummy encoding: {list(categorical_cols)}")
        df_eng = pd.get_dummies(df_eng, columns=categorical_cols, drop_first=True)
        print(f"After dummy encoding: {list(df_eng.columns)}")

    return df_eng


if __name__ == "__main__":
    debug_model()
