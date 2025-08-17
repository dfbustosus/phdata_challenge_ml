#!/usr/bin/env python3
"""
Comprehensive ML Training Pipeline for Housing Price Prediction.

This pipeline implements ML best practices including data cleaning, feature engineering,
model selection, hyperparameter tuning, and comprehensive evaluation.

Author: ML Engineering Team
Version: 1.0.0
Date: 2025-08-17
"""

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn import (
    ensemble, linear_model, model_selection, metrics, preprocessing, 
    pipeline, feature_selection
)

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'data_dir': 'data',
    'model_dir': 'model',
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'n_jobs': -1
}


class MLPipeline:
    """Comprehensive ML training pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        np.random.seed(config['random_state'])
        
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the training data."""
        print("📁 Loading and cleaning data...")
        
        # Load sales data
        sales_cols = ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
                     "floors", "sqft_above", "sqft_basement", "zipcode"]
        
        sales_data = pd.read_csv(
            Path(self.config['data_dir']) / "kc_house_data.csv",
            usecols=sales_cols, dtype={"zipcode": str}
        )
        
        # Load demographics
        demographics = pd.read_csv(
            Path(self.config['data_dir']) / "zipcode_demographics.csv",
            dtype={"zipcode": str}
        )
        
        # Merge data
        df = sales_data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
        
        print(f"   Initial shape: {df.shape}")
        
        # Data cleaning
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove outliers (conservative approach)
        df = df[(df['price'] >= df['price'].quantile(0.01)) & 
                (df['price'] <= df['price'].quantile(0.99))]
        
        # Domain-specific cleaning
        df = df[(df['bedrooms'] >= 0) & (df['bedrooms'] <= 20)]
        df = df[(df['bathrooms'] >= 0) & (df['bathrooms'] <= 15)]
        df = df[(df['sqft_living'] >= 100) & (df['sqft_living'] <= 20000)]
        df = df[(df['sqft_lot'] >= 100) & (df['sqft_lot'] <= 1000000)]
        df = df[(df['floors'] >= 1) & (df['floors'] <= 5)]
        
        # Remove invalid relationships
        valid_sqft = (df['sqft_living'] >= (df['sqft_above'] + df['sqft_basement']) * 0.8) & \
                    (df['sqft_living'] <= (df['sqft_above'] + df['sqft_basement']) * 1.2)
        df = df[valid_sqft]
        
        # Handle missing values with median imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        removed = initial_count - len(df)
        print(f"   Cleaned shape: {df.shape} ({removed} rows removed, {removed/initial_count*100:.1f}%)")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features."""
        print("⚙️ Engineering features...")
        
        df_eng = df.copy()
        
        # Ratio features
        df_eng['sqft_living_to_lot_ratio'] = df_eng['sqft_living'] / (df_eng['sqft_lot'] + 1)
        df_eng['bathroom_to_bedroom_ratio'] = df_eng['bathrooms'] / (df_eng['bedrooms'] + 1)
        df_eng['above_to_living_ratio'] = df_eng['sqft_above'] / (df_eng['sqft_living'] + 1)
        df_eng['basement_to_living_ratio'] = df_eng['sqft_basement'] / (df_eng['sqft_living'] + 1)
        
        # Room density
        df_eng['room_density'] = (df_eng['bedrooms'] + df_eng['bathrooms']) / (df_eng['sqft_living'] + 1)
        
        # Size categories
        df_eng['size_category'] = pd.cut(df_eng['sqft_living'], 
                                        bins=[0, 1000, 1500, 2000, 3000, float('inf')],
                                        labels=[1, 2, 3, 4, 5])
        
        # Log transforms for skewed features (EXCLUDE TARGET VARIABLE)
        skewed_features = ['sqft_living', 'sqft_lot']  # Removed 'price' to prevent target leakage
        for feature in skewed_features:
            if feature in df_eng.columns:
                df_eng[f'{feature}_log'] = np.log1p(df_eng[feature])
        
        print(f"   Features created: {len(df_eng.columns)} total features")
        return df_eng
    
    def get_models(self) -> Dict[str, Any]:
        """Get candidate models with parameter grids and proper regularization."""
        return {
            'random_forest': {
                'model': ensemble.RandomForestRegressor(
                    random_state=self.config['random_state'], n_jobs=self.config['n_jobs']
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, 15],  # More conservative depths
                    'min_samples_split': [5, 10],  # Higher min samples to prevent overfitting
                    'min_samples_leaf': [2, 4],  # Add min samples per leaf
                    'max_features': ['sqrt', 0.5]  # Limit features to reduce overfitting
                }
            },
            'gradient_boosting': {
                'model': ensemble.GradientBoostingRegressor(
                    random_state=self.config['random_state']
                ),
                'params': {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 4, 5],
                    'subsample': [0.8, 0.9],  # Regularization via subsampling
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 4]
                }
            },
            'hist_gradient_boosting': {
                'model': ensemble.HistGradientBoostingRegressor(
                    random_state=self.config['random_state']
                ),
                'params': {
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_iter': [100, 150],
                    'max_depth': [3, 4, 5],
                    'min_samples_leaf': [10, 20],  # Regularization
                    'l2_regularization': [0.0, 0.1, 1.0]  # L2 regularization
                }
            },
            'ridge': {
                'model': pipeline.Pipeline([
                    ('scaler', preprocessing.StandardScaler()),
                    ('regressor', linear_model.Ridge())
                ]),
                'params': {
                    'regressor__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]  # Stronger regularization
                }
            },
            'elastic_net': {
                'model': pipeline.Pipeline([
                    ('scaler', preprocessing.StandardScaler()),
                    ('regressor', linear_model.ElasticNet(random_state=self.config['random_state']))
                ]),
                'params': {
                    'regressor__alpha': [0.1, 1.0, 10.0],
                    'regressor__l1_ratio': [0.1, 0.5, 0.7, 0.9]  # Mix of L1 and L2
                }
            }
        }
    
    def select_best_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict[str, Any]]:
        """Select best model using cross-validation."""
        print("🎯 Selecting best model...")
        
        models = self.get_models()
        results = {}
        
        for name, model_config in models.items():
            print(f"   Evaluating {name}...")
            
            try:
                grid_search = model_selection.GridSearchCV(
                    model_config['model'], model_config['params'],
                    cv=self.config['cv_folds'], scoring='r2',
                    n_jobs=self.config['n_jobs'], verbose=0
                )
                
                grid_search.fit(X, y)
                
                results[name] = {
                    'best_score': grid_search.best_score_,
                    'best_params': grid_search.best_params_,
                    'best_model': grid_search.best_estimator_
                }
                
                print(f"      R² = {grid_search.best_score_:.4f}")
                
            except Exception as e:
                print(f"      Failed: {e}")
                continue
        
        if not results:
            raise ValueError("No models were successfully trained")
        
        best_name = max(results.keys(), key=lambda k: results[k]['best_score'])
        best_model = results[best_name]['best_model']
        
        print(f"   🏆 Best model: {best_name} (R² = {results[best_name]['best_score']:.4f})")
        
        return best_model, results
    
    def evaluate_model(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        print("📊 Evaluating final model...")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        evaluation = {
            'train_r2': metrics.r2_score(y_train, y_train_pred),
            'test_r2': metrics.r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)),
            'train_mae': metrics.mean_absolute_error(y_train, y_train_pred),
            'test_mae': metrics.mean_absolute_error(y_test, y_test_pred)
        }
        
        # Overfitting analysis
        evaluation['overfitting_r2'] = evaluation['train_r2'] - evaluation['test_r2']
        evaluation['rmse_ratio'] = evaluation['test_rmse'] / evaluation['train_rmse']
        
        print(f"   Train R²: {evaluation['train_r2']:.4f}")
        print(f"   Test R²: {evaluation['test_r2']:.4f}")
        print(f"   Test RMSE: ${evaluation['test_rmse']:,.0f}")
        print(f"   Overfitting (R² diff): {evaluation['overfitting_r2']:.4f}")
        
        return evaluation
    
    def save_model(self, model: Any, feature_names: List[str], evaluation: Dict[str, Any]) -> None:
        """Save model and metadata to versioned directory."""
        print("💾 Saving model artifacts...")
        
        # Use versioned directory structure
        model_dir = Path(self.config['model_dir']) / "v2"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_dir / "model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        # Save feature names
        with open(model_dir / "model_features.json", 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        # Save metadata
        metadata = {
            'version': '2.0.0',
            'model_type': type(model).__name__,
            'training_date': datetime.now().isoformat(),
            'evaluation_metrics': evaluation,
            'feature_count': len(feature_names),
            'config': self.config
        }
        
        with open(model_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Model saved to {model_dir}")
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the complete training pipeline."""
        print("🚀 Starting ML Training Pipeline")
        print("=" * 50)
        
        # 1. Load and clean data
        df = self.load_and_clean_data()
        
        # 2. Feature engineering
        df_eng = self.engineer_features(df)
        
        # 3. Prepare features and target
        y = df_eng['price'].copy()
        X = df_eng.drop(columns=['price'])
        
        # Handle any remaining categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        feature_names = list(X.columns)
        
        print(f"   Final feature set: {len(feature_names)} features")
        
        # 4. Train/test split
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        # 5. Model selection
        best_model, model_results = self.select_best_model(X_train, y_train)
        
        # 6. Final evaluation
        evaluation = self.evaluate_model(best_model, X_train, X_test, y_train, y_test)
        
        # 7. Save model
        self.save_model(best_model, feature_names, evaluation)
        
        # 8. Generate report
        report = {
            'data_shape': df.shape,
            'feature_count': len(feature_names),
            'model_results': model_results,
            'final_evaluation': evaluation,
            'training_completed': datetime.now().isoformat()
        }
        
        print("\n" + "=" * 50)
        print("✅ Training pipeline completed successfully!")
        
        # Assessment
        test_r2 = evaluation['test_r2']
        overfitting = evaluation['overfitting_r2']
        
        if test_r2 > 0.8 and overfitting < 0.05:
            assessment = "EXCELLENT"
        elif test_r2 > 0.75 and overfitting < 0.1:
            assessment = "GOOD"
        elif test_r2 > 0.7 and overfitting < 0.15:
            assessment = "FAIR"
        else:
            assessment = "NEEDS IMPROVEMENT"
        
        print(f"🎯 Model Assessment: {assessment}")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   Overfitting: {overfitting:.4f}")
        
        return report


def main():
    """Main function to run the training pipeline."""
    try:
        pipeline = MLPipeline(CONFIG)
        report = pipeline.run_pipeline()
        
        # Save training report
        with open("training_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("\n📋 Training report saved to training_report.json")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
