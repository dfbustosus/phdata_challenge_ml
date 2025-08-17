#!/usr/bin/env python3
"""
Test script for Housing Price Prediction API.

This script demonstrates the API functionality by submitting examples from
data/future_unseen_examples.csv to both the main prediction endpoint and
the minimal features endpoint.

Author: ML Engineering Team
Version: 1.0.0
Date: 2025-08-17
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class APITester:
    """Test client for the Housing Price Prediction API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API tester.
        
        Args:
            base_url: Base URL of the API service
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set timeout
        self.timeout = 30
    
    def wait_for_service(self, max_attempts: int = 30, delay: float = 2.0) -> bool:
        """Wait for the API service to become available.
        
        Args:
            max_attempts: Maximum number of attempts
            delay: Delay between attempts in seconds
            
        Returns:
            True if service is available, False otherwise
        """
        print(f"Waiting for API service at {self.base_url}...")
        
        for attempt in range(max_attempts):
            try:
                response = self.session.get(
                    f"{self.base_url}/health",
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") == "healthy":
                        print("✅ API service is healthy and ready")
                        return True
                    else:
                        print(f"⚠️  API service not ready: {health_data}")
                        
            except requests.exceptions.RequestException as e:
                print(f"⏳ Attempt {attempt + 1}/{max_attempts}: {e}")
            
            if attempt < max_attempts - 1:
                time.sleep(delay)
        
        print("❌ API service is not available")
        return False
    
    def test_health_endpoint(self) -> bool:
        """Test the health endpoint.
        
        Returns:
            True if test passes, False otherwise
        """
        print("\n🔍 Testing health endpoint...")
        
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_model_info_endpoint(self) -> bool:
        """Test the model info endpoint.
        
        Returns:
            True if test passes, False otherwise
        """
        print("\n🔍 Testing model info endpoint...")
        
        try:
            response = self.session.get(
                f"{self.base_url}/v1/model-info",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model info retrieved successfully:")
                print(f"   Model Type: {data.get('model_type')}")
                print(f"   Model Version: {data.get('model_version')}")
                print(f"   Number of Features: {data.get('n_features')}")
                print(f"   Number of Zipcodes: {data.get('n_zipcodes')}")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Model info error: {e}")
            return False
    
    def load_test_data(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load test data from CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of records for testing
        """
        print(f"\n📁 Loading test data from {csv_path}...")
        
        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} records")
        print(f"   Columns: {list(df.columns)}")
        
        # Convert to list of dictionaries
        records = df.to_dict('records')
        
        # Convert zipcode to string if it's not already
        for record in records:
            if 'zipcode' in record:
                record['zipcode'] = str(int(record['zipcode']))
        
        return records
    
    def prepare_main_prediction_data(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare data for the main prediction endpoint.
        
        Args:
            records: Raw records from CSV
            
        Returns:
            Filtered records with only required house features
        """
        required_fields = [
            "bedrooms", "bathrooms", "sqft_living", "sqft_lot", 
            "floors", "sqft_above", "sqft_basement", "zipcode"
        ]
        
        prepared_records = []
        for record in records:
            prepared_record = {}
            for field in required_fields:
                if field in record:
                    prepared_record[field] = record[field]
                else:
                    print(f"⚠️  Missing field {field} in record, skipping...")
                    break
            else:
                prepared_records.append(prepared_record)
        
        print(f"   Prepared {len(prepared_records)} records for main prediction")
        return prepared_records
    
    def prepare_minimal_prediction_data(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare data for the minimal prediction endpoint.
        
        Args:
            records: Raw records from CSV
            
        Returns:
            Filtered records with only minimal required features
        """
        required_fields = ["bedrooms", "bathrooms", "sqft_living", "zipcode"]
        
        prepared_records = []
        for record in records:
            prepared_record = {}
            for field in required_fields:
                if field in record:
                    prepared_record[field] = record[field]
                else:
                    print(f"⚠️  Missing field {field} in record, skipping...")
                    break
            else:
                prepared_records.append(prepared_record)
        
        print(f"   Prepared {len(prepared_records)} records for minimal prediction")
        return prepared_records
    
    def test_main_prediction(self, records: List[Dict[str, Any]], batch_size: int = 5) -> bool:
        """Test the main prediction endpoint.
        
        Args:
            records: Test records
            batch_size: Number of records to send per request
            
        Returns:
            True if test passes, False otherwise
        """
        print(f"\n🔍 Testing main prediction endpoint with {len(records)} records...")
        
        try:
            # Test with first batch of records
            test_records = records[:batch_size]
            
            payload = {"records": test_records}
            
            print(f"   Sending {len(test_records)} records...")
            response = self.session.post(
                f"{self.base_url}/v1/predict",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get("predictions", [])
                
                print(f"✅ Main prediction successful:")
                print(f"   Processed: {data.get('n_records')} records")
                print(f"   Processing time: {data.get('processing_time_ms'):.2f}ms")
                print(f"   Model type: {data.get('model_type')}")
                print(f"   Model version: {data.get('model_version')}")
                
                # Show first few predictions
                for i, pred in enumerate(predictions[:3]):
                    print(f"   Prediction {i+1}: ${pred['prediction']:,.2f} (zipcode: {pred['zipcode']})")
                
                if len(predictions) > 3:
                    print(f"   ... and {len(predictions) - 3} more predictions")
                
                return True
            else:
                print(f"❌ Main prediction failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Main prediction error: {e}")
            return False
    
    def test_minimal_prediction(self, records: List[Dict[str, Any]], batch_size: int = 5) -> bool:
        """Test the minimal prediction endpoint.
        
        Args:
            records: Test records
            batch_size: Number of records to send per request
            
        Returns:
            True if test passes, False otherwise
        """
        print(f"\n🔍 Testing minimal prediction endpoint with {len(records)} records...")
        
        try:
            # Test with first batch of records
            test_records = records[:batch_size]
            
            payload = {"records": test_records}
            
            print(f"   Sending {len(test_records)} records...")
            response = self.session.post(
                f"{self.base_url}/v1/predict-minimal",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get("predictions", [])
                
                print(f"✅ Minimal prediction successful:")
                print(f"   Processed: {data.get('n_records')} records")
                print(f"   Processing time: {data.get('processing_time_ms'):.2f}ms")
                print(f"   Model type: {data.get('model_type')}")
                print(f"   Model version: {data.get('model_version')}")
                
                # Show first few predictions
                for i, pred in enumerate(predictions[:3]):
                    print(f"   Prediction {i+1}: ${pred['prediction']:,.2f} (zipcode: {pred['zipcode']})")
                
                if len(predictions) > 3:
                    print(f"   ... and {len(predictions) - 3} more predictions")
                
                return True
            else:
                print(f"❌ Minimal prediction failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Minimal prediction error: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test API error handling with invalid data.
        
        Returns:
            True if error handling works correctly, False otherwise
        """
        print("\n🔍 Testing error handling...")
        
        test_cases = [
            {
                "name": "Empty records",
                "payload": {"records": []},
                "expected_status": 422
            },
            {
                "name": "Missing zipcode",
                "payload": {"records": [{"bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1500}]},
                "expected_status": 422
            },
            {
                "name": "Invalid zipcode",
                "payload": {"records": [{"bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1500, "zipcode": "invalid"}]},
                "expected_status": 422
            },
            {
                "name": "Negative values",
                "payload": {"records": [{"bedrooms": -1, "bathrooms": 2.0, "sqft_living": 1500, "zipcode": "98001"}]},
                "expected_status": 422
            }
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/predict",
                    json=test_case["payload"],
                    timeout=self.timeout
                )
                
                if response.status_code == test_case["expected_status"]:
                    print(f"   ✅ {test_case['name']}: Correctly returned {response.status_code}")
                else:
                    print(f"   ❌ {test_case['name']}: Expected {test_case['expected_status']}, got {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ {test_case['name']}: Error - {e}")
                all_passed = False
        
        return all_passed
    
    def run_comprehensive_test(self, csv_path: str) -> bool:
        """Run comprehensive API tests.
        
        Args:
            csv_path: Path to test data CSV
            
        Returns:
            True if all tests pass, False otherwise
        """
        print("🚀 Starting comprehensive API tests...")
        print("=" * 60)
        
        # Wait for service to be ready
        if not self.wait_for_service():
            return False
        
        # Load test data
        try:
            raw_records = self.load_test_data(csv_path)
        except Exception as e:
            print(f"❌ Failed to load test data: {e}")
            return False
        
        # Run all tests
        tests = [
            ("Health Endpoint", lambda: self.test_health_endpoint()),
            ("Model Info Endpoint", lambda: self.test_model_info_endpoint()),
            ("Main Prediction", lambda: self.test_main_prediction(
                self.prepare_main_prediction_data(raw_records)
            )),
            ("Minimal Prediction", lambda: self.test_minimal_prediction(
                self.prepare_minimal_prediction_data(raw_records)
            )),
            ("Error Handling", lambda: self.test_error_handling())
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:<25} {status}")
            if result:
                passed += 1
        
        print("-" * 60)
        print(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! API is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Please check the API implementation.")
            return False


def main():
    """Main function to run the API tests."""
    # Configuration
    api_url = "http://localhost:8000"
    test_data_path = "data/future_unseen_examples.csv"
    
    # Check if test data exists
    if not Path(test_data_path).exists():
        print(f"❌ Test data file not found: {test_data_path}")
        print("Please ensure the file exists and run the script from the project root directory.")
        return False
    
    # Run tests
    tester = APITester(api_url)
    success = tester.run_comprehensive_test(test_data_path)
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
