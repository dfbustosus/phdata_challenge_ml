# Housing Price Prediction API Documentation

## Overview

The Housing Price Prediction API provides RESTful endpoints for predicting housing prices using machine learning models. The API supports multiple model versions and automatically merges demographic data based on zipcode.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required for development. For production deployment, implement proper authentication mechanisms as outlined in the deployment guide.

## Endpoints

### 1. Health Check

Check the API service health and model loading status.

**Endpoint:** `GET /health`

**curl Example:**
```bash
curl -X GET "http://localhost:8000/health" \
  -H "Accept: application/json"
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "model_loaded": true,
  "demographics_loaded": true
}
```

### 2. Model Information

Get information about available model versions and features.

**Endpoint:** `GET /v1/model-info`

**Parameters:**
- `model_version` (optional): Model version to query (default: v2)

**curl Examples:**

Get default model info:
```bash
curl -X GET "http://localhost:8000/v1/model-info" \
  -H "Accept: application/json"
```

Get specific model version info:
```bash
curl -X GET "http://localhost:8000/v1/model-info?model_version=v1" \
  -H "Accept: application/json"
```

**Response:**
```json
{
  "model_version": "2.0.0",
  "model_type": "GradientBoostingRegressor",
  "feature_count": 44,
  "features": ["bedrooms", "bathrooms", "sqft_living", "..."],
  "available_versions": ["v1", "v2"],
  "default_version": "v2",
  "api_version": "2.0.0",
  "endpoints": ["/v1/predict", "/v1/predict-minimal", "/v1/model-info", "/health"]
}
```

### 3. Main Prediction Endpoint

Predict housing prices using complete house features. Demographic data is merged automatically based on zipcode.

**Endpoint:** `POST /v1/predict`

**Parameters:**
- `model_version` (optional): Model version to use (default: v2)

**Request Body:**
```json
{
  "records": [
    {
      "bedrooms": 3,
      "bathrooms": 2.5,
      "sqft_living": 2000,
      "sqft_lot": 8000,
      "floors": 2.0,
      "sqft_above": 2000,
      "sqft_basement": 0,
      "zipcode": "98001"
    }
  ]
}
```

**curl Example:**
```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "records": [
      {
        "bedrooms": 3,
        "bathrooms": 2.5,
        "sqft_living": 2000,
        "sqft_lot": 8000,
        "floors": 2.0,
        "sqft_above": 2000,
        "sqft_basement": 0,
        "zipcode": "98001"
      }
    ]
  }'
```

**curl Example with specific model version:**
```bash
curl -X POST "http://localhost:8000/v1/predict?model_version=v1" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "records": [
      {
        "bedrooms": 4,
        "bathrooms": 3.0,
        "sqft_living": 2500,
        "sqft_lot": 10000,
        "floors": 2.0,
        "sqft_above": 2500,
        "sqft_basement": 0,
        "zipcode": "98004"
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [450000.0],
  "model_version": "2.0.0",
  "model_type": "GradientBoostingRegressor",
  "n_records": 1,
  "feature_count": 44
}
```

### 4. Minimal Features Prediction Endpoint

Predict housing prices using only essential features. Missing features are filled with intelligent defaults.

**Endpoint:** `POST /v1/predict-minimal`

**Parameters:**
- `model_version` (optional): Model version to use (default: v2)

**Request Body:**
```json
{
  "records": [
    {
      "bedrooms": 3,
      "bathrooms": 2.0,
      "sqft_living": 1800,
      "zipcode": "98001"
    }
  ]
}
```

**curl Example:**
```bash
curl -X POST "http://localhost:8000/v1/predict-minimal" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "records": [
      {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1800,
        "zipcode": "98001"
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [420000.0],
  "model_version": "2.0.0",
  "model_type": "GradientBoostingRegressor",
  "n_records": 1,
  "feature_count": 44
}
```

## Batch Predictions

Both prediction endpoints support batch processing. Simply include multiple records in the request:

**curl Example (Batch):**
```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "records": [
      {
        "bedrooms": 3,
        "bathrooms": 2.5,
        "sqft_living": 2000,
        "sqft_lot": 8000,
        "floors": 2.0,
        "sqft_above": 2000,
        "sqft_basement": 0,
        "zipcode": "98001"
      },
      {
        "bedrooms": 4,
        "bathrooms": 3.0,
        "sqft_living": 2800,
        "sqft_lot": 12000,
        "floors": 2.0,
        "sqft_above": 2800,
        "sqft_basement": 0,
        "zipcode": "98004"
      }
    ]
  }'
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

### Common Error Responses

**400 Bad Request - Invalid Input:**
```json
{
  "detail": "Invalid input data: zipcode must contain only digits"
}
```

**503 Service Unavailable - Model Not Available:**
```json
{
  "detail": "Model service unavailable: Model v3 not found at model/v3/model.pkl"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Prediction failed: Internal server error"
}
```

## Input Validation

### House Features Validation

- `bedrooms`: Integer, 0-50
- `bathrooms`: Float, 0-20
- `sqft_living`: Integer, 100-50,000
- `sqft_lot`: Integer, 100-1,000,000
- `floors`: Float, 1-10
- `sqft_above`: Integer, 0-50,000
- `sqft_basement`: Integer, 0-50,000
- `zipcode`: String, exactly 5 digits

### Batch Limits

- Minimum records per request: 1
- Maximum records per request: 1000

## Model Versions

The API supports multiple model versions:

- **v1**: Original KNeighborsRegressor model
- **v2**: Improved GradientBoostingRegressor with regularization (default)

Specify the model version using the `model_version` query parameter.

## Rate Limiting

For production deployment, implement appropriate rate limiting. See the deployment guide for recommendations.

## Testing the API

### Using the provided test script:

```bash
# Run comprehensive API tests
python tests/test_api.py

# Test with specific examples
python scripts/client.py --api http://localhost:8000/v1/predict --csv data/future_unseen_examples.csv --n 5
```

### Quick health check:

```bash
curl -f http://localhost:8000/health || echo "API is down"
```

## Performance Considerations

- Model loading is cached for performance
- Demographic data is loaded once and cached
- Response times typically < 100ms for single predictions
- Batch processing is more efficient for multiple predictions

## Security Notes

- Input validation prevents malicious data injection
- Demographic data is never exposed in API responses
- For production, implement proper authentication and HTTPS
- See deployment guide for comprehensive security measures

## Support

For issues or questions, refer to the technical documentation and deployment guides in the `docs/` directory.
