# Housing Price Prediction API - Deployment & Scaling Guide

**Author:** David BU
**Version:** 1.0.0  
**Date:** 2025-08-17

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Deployment Strategies](#deployment-strategies)
3. [Scaling Considerations](#scaling-considerations)
4. [Model Update Strategies](#model-update-strategies)
5. [Infrastructure Requirements](#infrastructure-requirements)
6. [Security Considerations](#security-considerations)
7. [Monitoring & Observability](#monitoring--observability)
8. [Performance Optimization](#performance-optimization)

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   API Gateway   │────│   FastAPI App   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Model Cache   │────│  Demographics   │
                       │   (Redis)       │    │   Database      │
                       └─────────────────┘    └─────────────────┘
```

### Core Design Principles

1. **Stateless Services**: API instances are stateless for horizontal scaling
2. **Caching Strategy**: Model and demographics data cached in memory/Redis
3. **Separation of Concerns**: Model inference separated from data management
4. **Graceful Degradation**: Service continues with reduced functionality if components fail
5. **Zero-Downtime Deployments**: Blue-green deployment strategy for updates

## Deployment Strategies

### 1. Container-Based Deployment (Recommended)

#### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy application code
COPY src/ ./src/
COPY model/ ./model/
COPY data/zipcode_demographics.csv ./data/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.housing_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_DIR=/app/model
      - DEMOGRAPHICS_CSV=/app/data/zipcode_demographics.csv
    volumes:
      - ./model:/app/model:ro
      - ./data:/app/data:ro
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api
    restart: unless-stopped

volumes:
  redis_data:
```

### 2. Kubernetes Deployment (Production)

#### Deployment Manifest

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: housing-api
  labels:
    app: housing-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: housing-api
  template:
    metadata:
      labels:
        app: housing-api
    spec:
      containers:
      - name: api
        image: housing-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_DIR
          value: "/app/model"
        - name: DEMOGRAPHICS_CSV
          value: "/app/data/zipcode_demographics.csv"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-volume
          mountPath: /app/model
          readOnly: true
      volumes:
      - name: model-volume
        configMap:
          name: model-artifacts

---
apiVersion: v1
kind: Service
metadata:
  name: housing-api-service
spec:
  selector:
    app: housing-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: housing-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: housing-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Scaling Considerations

### Horizontal Scaling

#### Auto-Scaling Metrics
- **CPU Utilization**: Target 70% average
- **Memory Utilization**: Target 80% average
- **Request Rate**: Scale up when >100 RPS per instance
- **Response Time**: Scale up when P95 latency >500ms

#### Load Balancing Strategy
```nginx
# nginx.conf
upstream housing_api {
    least_conn;
    server api1:8000 max_fails=3 fail_timeout=30s;
    server api2:8000 max_fails=3 fail_timeout=30s;
    server api3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://housing_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://housing_api;
    }
}
```

### Vertical Scaling

#### Resource Optimization
- **Memory**: 1GB per instance (model + demographics + overhead)
- **CPU**: 2 cores per instance for optimal performance
- **Storage**: Minimal (stateless design)

### Caching Strategy

#### Model Caching
```python
# Enhanced model service with Redis caching
import redis
import pickle

class CachedModelService:
    def __init__(self):
        self.redis_client = redis.Redis(host='redis', port=6379, db=0)
        self.cache_ttl = 3600  # 1 hour
        
    def load_model_and_features(self, model_dir: Path):
        cache_key = f"model:{model_dir}"
        
        # Try cache first
        cached_data = self.redis_client.get(cache_key)
        if cached_data:
            return pickle.loads(cached_data)
            
        # Load from disk and cache
        model, features = self._load_from_disk(model_dir)
        self.redis_client.setex(
            cache_key, 
            self.cache_ttl, 
            pickle.dumps((model, features))
        )
        
        return model, features
```

## Model Update Strategies

### 1. Blue-Green Deployment

#### Process Flow
1. **Prepare Green Environment**: Deploy new model to green environment
2. **Health Checks**: Validate green environment health
3. **Traffic Switch**: Gradually shift traffic from blue to green
4. **Monitor**: Watch metrics during transition
5. **Rollback Plan**: Keep blue environment ready for quick rollback

#### Implementation Script
```bash
#!/bin/bash
# deploy-model.sh

MODEL_VERSION=$1
ENVIRONMENT=${2:-production}

echo "Deploying model version $MODEL_VERSION to $ENVIRONMENT"

# Build new image with updated model
docker build -t housing-api:$MODEL_VERSION .

# Deploy to green environment
kubectl set image deployment/housing-api-green api=housing-api:$MODEL_VERSION

# Wait for rollout
kubectl rollout status deployment/housing-api-green --timeout=300s

# Health check
for i in {1..10}; do
    if curl -f http://green-housing-api/health; then
        echo "Green environment healthy"
        break
    fi
    sleep 10
done

# Switch traffic
kubectl patch service housing-api-service -p '{"spec":{"selector":{"version":"green"}}}'

echo "Deployment completed successfully"
```

### 2. Canary Deployment

#### Traffic Splitting Configuration
```yaml
# Istio VirtualService for canary deployment
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: housing-api-canary
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: housing-api-service
        subset: v2
  - route:
    - destination:
        host: housing-api-service
        subset: v1
      weight: 90
    - destination:
        host: housing-api-service
        subset: v2
      weight: 10
```

### 3. A/B Testing for Model Performance

#### Model Comparison Framework
```python
class ModelComparisonService:
    def __init__(self):
        self.model_a = self.load_model("model_a")
        self.model_b = self.load_model("model_b")
        self.metrics_collector = MetricsCollector()
        
    def predict_with_comparison(self, features, user_id):
        # Determine which model to use (hash-based splitting)
        use_model_b = hash(user_id) % 100 < 10  # 10% traffic to model B
        
        if use_model_b:
            prediction = self.model_b.predict(features)
            self.metrics_collector.record("model_b", prediction, user_id)
        else:
            prediction = self.model_a.predict(features)
            self.metrics_collector.record("model_a", prediction, user_id)
            
        return prediction
```

## Infrastructure Requirements

### Minimum Production Setup

#### Single Region Deployment
- **Load Balancer**: 1 instance (AWS ALB/GCP Load Balancer)
- **API Instances**: 3 instances (2 vCPU, 4GB RAM each)
- **Cache**: Redis cluster (3 nodes, 2GB RAM each)
- **Database**: PostgreSQL for demographics (if migrated from CSV)
- **Monitoring**: Prometheus + Grafana stack

#### Multi-Region Deployment
- **Regions**: 2-3 regions for redundancy
- **CDN**: CloudFront/CloudFlare for static assets
- **Database Replication**: Read replicas in each region
- **Cross-Region Load Balancing**: Route 53/Cloud DNS

### Cost Estimation (AWS)

| Component | Instance Type | Count | Monthly Cost |
|-----------|---------------|-------|--------------|
| ALB | - | 1 | $20 |
| API Instances | t3.medium | 3 | $95 |
| Redis | t3.micro | 3 | $30 |
| RDS (Demographics) | t3.micro | 1 | $15 |
| CloudWatch | - | - | $10 |
| **Total** | | | **$170/month** |

## Security Considerations

### API Security

#### Authentication & Authorization
```python
# Enhanced security middleware
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(
            credentials.credentials, 
            SECRET_KEY, 
            algorithms=["HS256"]
        )
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )

@app.post("/v1/predict")
async def predict(
    req: PredictRequest,
    user: dict = Depends(verify_token)
):
    # Implementation with user context
    pass
```

#### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/v1/predict")
@limiter.limit("100/minute")
async def predict(request: Request, req: PredictRequest):
    # Implementation
    pass
```

### Infrastructure Security

#### Network Security
- **VPC**: Isolated network environment
- **Security Groups**: Restrictive ingress/egress rules
- **WAF**: Web Application Firewall for DDoS protection
- **TLS**: End-to-end encryption (TLS 1.3)

#### Secrets Management
```yaml
# Kubernetes secrets
apiVersion: v1
kind: Secret
metadata:
  name: housing-api-secrets
type: Opaque
data:
  jwt-secret: <base64-encoded-secret>
  redis-password: <base64-encoded-password>
```

## Monitoring & Observability

### Metrics Collection

#### Application Metrics
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
prediction_requests = Counter('prediction_requests_total', 'Total prediction requests')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')
model_load_time = Gauge('model_load_time_seconds', 'Model load time')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    prediction_duration.observe(duration)
    prediction_requests.inc()
    
    return response
```

#### Infrastructure Monitoring
- **CPU/Memory Usage**: Per instance and aggregate
- **Network I/O**: Request/response metrics
- **Disk Usage**: Model storage monitoring
- **Cache Hit Rate**: Redis performance metrics

### Alerting Rules

#### Critical Alerts
```yaml
# Prometheus alerting rules
groups:
- name: housing-api
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    annotations:
      summary: "High error rate detected"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(prediction_duration_seconds_bucket[5m])) > 1.0
    for: 5m
    annotations:
      summary: "High prediction latency detected"
      
  - alert: ModelLoadFailure
    expr: model_load_time_seconds == 0
    for: 1m
    annotations:
      summary: "Model failed to load"
```

### Logging Strategy

#### Structured Logging
```python
import structlog

logger = structlog.get_logger()

@app.post("/v1/predict")
async def predict(req: PredictRequest):
    logger.info(
        "prediction_request",
        n_records=len(req.records),
        user_id=get_user_id(),
        timestamp=datetime.utcnow().isoformat()
    )
    
    try:
        result = make_prediction(req)
        
        logger.info(
            "prediction_success",
            n_records=len(req.records),
            processing_time_ms=result.processing_time_ms
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "prediction_error",
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

## Performance Optimization

### Response Time Optimization

#### Caching Strategy
1. **Model Caching**: Keep model in memory across requests
2. **Demographics Caching**: Cache demographic data by zipcode
3. **Response Caching**: Cache predictions for identical inputs (short TTL)

#### Async Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/v1/predict")
async def predict(req: PredictRequest):
    # Offload CPU-intensive prediction to thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        make_prediction_sync, 
        req
    )
    return result
```

### Memory Optimization

#### Model Quantization
```python
# Model compression for production
import joblib
from sklearn.externals import joblib

# Save compressed model
joblib.dump(model, 'model.pkl', compress=3)

# Memory-efficient loading
model = joblib.load('model.pkl', mmap_mode='r')
```

### Database Optimization

#### Demographics Data Migration
```sql
-- PostgreSQL schema for demographics
CREATE TABLE zipcode_demographics (
    zipcode VARCHAR(5) PRIMARY KEY,
    ppltn_qty INTEGER,
    urbn_ppltn_qty INTEGER,
    -- ... other demographic fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_zipcode_demographics_zipcode ON zipcode_demographics(zipcode);
```

## Disaster Recovery

### Backup Strategy
- **Model Artifacts**: Versioned storage in S3/GCS
- **Demographics Data**: Daily backups
- **Configuration**: Infrastructure as Code (Terraform/CloudFormation)

### Recovery Procedures
1. **Service Outage**: Auto-scaling and health checks
2. **Data Corruption**: Restore from latest backup
3. **Region Failure**: Failover to secondary region
4. **Model Rollback**: Quick deployment of previous version

## Conclusion

This deployment guide provides a comprehensive framework for scaling the Housing Price Prediction API from development to production. The strategies outlined ensure:

- **High Availability**: 99.9% uptime through redundancy
- **Scalability**: Handle 1000+ RPS with auto-scaling
- **Security**: Enterprise-grade security controls
- **Observability**: Complete monitoring and alerting
- **Maintainability**: Zero-downtime deployments and updates

For implementation, start with the containerized deployment and gradually adopt more advanced strategies based on traffic and business requirements.
