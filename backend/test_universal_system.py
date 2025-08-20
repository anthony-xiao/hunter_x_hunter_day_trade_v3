#!/usr/bin/env python3
"""
Universal Training System Comprehensive Test Suite

This script validates all components of the Universal Training System implementation:
1. Universal training API endpoints
2. Universal model initialization and loading
3. Mode switching functionality
4. Data pipeline batch processing
5. Signal generation with universal models
6. Backward compatibility
7. Performance validation
"""

import asyncio
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import os

# Add backend directory to path
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

from data.data_pipeline import DataPipeline
from ml.model_trainer import ModelTrainer
from trading.signal_generator import SignalGenerator

class UniversalSystemTester:
    """Comprehensive test suite for Universal Training System"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_symbols = ["AAPL", "TSLA", "NVDA"]
        self.test_results = {
            "api_endpoints": {},
            "model_initialization": {},
            "mode_switching": {},
            "data_pipeline": {},
            "signal_generation": {},
            "backward_compatibility": {},
            "performance_validation": {},
            "overall_status": "PENDING"
        }
        self.start_time = time.time()
        
    def log_test(self, category: str, test_name: str, status: str, details: str = ""):
        """Log test results"""
        timestamp = datetime.now().isoformat()
        self.test_results[category][test_name] = {
            "status": status,
            "details": details,
            "timestamp": timestamp
        }
        print(f"[{timestamp}] {category}.{test_name}: {status} - {details}")
    
    async def test_api_endpoints(self):
        """Test 1: Universal training API endpoints"""
        print("\n=== Testing Universal Training API Endpoints ===")
        
        # Test 1.1: Universal training endpoint availability
        try:
            response = requests.get(f"{self.base_url}/docs")
            if response.status_code == 200:
                self.log_test("api_endpoints", "docs_available", "PASS", "API documentation accessible")
            else:
                self.log_test("api_endpoints", "docs_available", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("api_endpoints", "docs_available", "FAIL", str(e))
        
        # Test 1.2: Universal training job creation
        try:
            training_config = {
                "symbols": self.test_symbols,
                "training_config": {
                    "base_model_epochs": 5,
                    "fine_tune_epochs": 3,
                    "ensemble_optimization": True,
                    "progressive_training": True
                },
                "data_config": {
                    "training_window_months": 6,
                    "validation_window_months": 2,
                    "feature_set": "reduced"
                }
            }
            
            response = requests.post(
                f"{self.base_url}/models/universal/train",
                json=training_config,
                timeout=30
            )
            
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data.get("job_id")
                self.log_test("api_endpoints", "universal_training_start", "PASS", f"Job ID: {job_id}")
                
                # Test 1.3: Training status monitoring
                await asyncio.sleep(5)  # Wait for job to start
                status_response = requests.get(f"{self.base_url}/models/universal/train/status/{job_id}")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    self.log_test("api_endpoints", "status_monitoring", "PASS", f"Status: {status_data.get('status')}")
                else:
                    self.log_test("api_endpoints", "status_monitoring", "FAIL", f"Status: {status_response.status_code}")
                    
                # Test 1.4: Job listing
                jobs_response = requests.get(f"{self.base_url}/models/universal/train/jobs")
                if jobs_response.status_code == 200:
                    jobs_data = jobs_response.json()
                    self.log_test("api_endpoints", "job_listing", "PASS", f"Found {len(jobs_data)} jobs")
                else:
                    self.log_test("api_endpoints", "job_listing", "FAIL", f"Status: {jobs_response.status_code}")
                    
            else:
                self.log_test("api_endpoints", "universal_training_start", "FAIL", f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_test("api_endpoints", "universal_training_start", "FAIL", str(e))
    
    async def test_model_initialization(self):
        """Test 2: Universal model initialization and loading"""
        print("\n=== Testing Universal Model Initialization ===")
        
        try:
            # Test 2.1: Model trainer initialization
            model_trainer = ModelTrainer()
            self.log_test("model_initialization", "trainer_init", "PASS", "ModelTrainer initialized")
            
            # Test 2.2: Universal training capability check
            if hasattr(model_trainer, 'initialize_universal_training'):
                self.log_test("model_initialization", "universal_methods", "PASS", "Universal training methods available")
            else:
                self.log_test("model_initialization", "universal_methods", "FAIL", "Universal training methods missing")
            
            # Test 2.3: Symbol embedding initialization
            if hasattr(model_trainer, 'symbol_to_id'):
                self.log_test("model_initialization", "symbol_embedding", "PASS", "Symbol embedding support available")
            else:
                self.log_test("model_initialization", "symbol_embedding", "FAIL", "Symbol embedding support missing")
                
        except Exception as e:
            self.log_test("model_initialization", "trainer_init", "FAIL", str(e))
    
    async def test_mode_switching(self):
        """Test 3: Universal vs symbol-specific mode switching"""
        print("\n=== Testing Mode Switching Functionality ===")
        
        try:
            # Test 3.1: Check universal mode status
            response = requests.get(f"{self.base_url}/models/universal/status")
            if response.status_code == 200:
                status_data = response.json()
                current_mode = status_data.get("is_universal_mode", False)
                self.log_test("mode_switching", "status_check", "PASS", f"Current mode: {'Universal' if current_mode else 'Symbol-specific'}")
                
                # Test 3.2: Enable universal mode
                enable_response = requests.post(f"{self.base_url}/models/universal/enable")
                if enable_response.status_code == 200:
                    self.log_test("mode_switching", "enable_universal", "PASS", "Universal mode enabled")
                    
                    # Test 3.3: Verify mode change
                    await asyncio.sleep(2)
                    verify_response = requests.get(f"{self.base_url}/models/universal/status")
                    if verify_response.status_code == 200:
                        verify_data = verify_response.json()
                        if verify_data.get("is_universal_mode", False):
                            self.log_test("mode_switching", "mode_verification", "PASS", "Mode switch verified")
                        else:
                            self.log_test("mode_switching", "mode_verification", "FAIL", "Mode switch not reflected")
                    
                    # Test 3.4: Disable universal mode
                    disable_response = requests.post(f"{self.base_url}/models/universal/disable")
                    if disable_response.status_code == 200:
                        self.log_test("mode_switching", "disable_universal", "PASS", "Universal mode disabled")
                    else:
                        self.log_test("mode_switching", "disable_universal", "FAIL", f"Status: {disable_response.status_code}")
                        
                else:
                    self.log_test("mode_switching", "enable_universal", "FAIL", f"Status: {enable_response.status_code}")
            else:
                self.log_test("mode_switching", "status_check", "FAIL", f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_test("mode_switching", "status_check", "FAIL", str(e))
    
    async def test_data_pipeline(self):
        """Test 4: Data pipeline batch processing for multiple symbols"""
        print("\n=== Testing Data Pipeline Batch Processing ===")
        
        try:
            # Test 4.1: Data pipeline initialization
            data_pipeline = DataPipeline()
            self.log_test("data_pipeline", "initialization", "PASS", "DataPipeline initialized")
            
            # Test 4.2: Universal data loading capability
            if hasattr(data_pipeline, 'load_universal_data'):
                self.log_test("data_pipeline", "universal_methods", "PASS", "Universal data methods available")
                
                # Test 4.3: Batch data loading
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                try:
                    # Test with a small dataset first
                    test_data = data_pipeline.load_universal_data(
                        symbols=self.test_symbols[:2],  # Test with 2 symbols
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                    
                    if test_data is not None and len(test_data) > 0:
                        self.log_test("data_pipeline", "batch_loading", "PASS", f"Loaded {len(test_data)} records")
                        
                        # Test 4.4: Universal feature engineering
                        if hasattr(data_pipeline, 'engineer_universal_features'):
                            features = data_pipeline.engineer_universal_features(test_data)
                            if features is not None:
                                self.log_test("data_pipeline", "universal_features", "PASS", f"Generated {len(features.columns)} features")
                            else:
                                self.log_test("data_pipeline", "universal_features", "FAIL", "Feature engineering returned None")
                        else:
                            self.log_test("data_pipeline", "universal_features", "FAIL", "Universal feature engineering not available")
                    else:
                        self.log_test("data_pipeline", "batch_loading", "FAIL", "No data loaded")
                        
                except Exception as e:
                    self.log_test("data_pipeline", "batch_loading", "FAIL", str(e))
            else:
                self.log_test("data_pipeline", "universal_methods", "FAIL", "Universal data methods missing")
                
        except Exception as e:
            self.log_test("data_pipeline", "initialization", "FAIL", str(e))
    
    async def test_signal_generation(self):
        """Test 5: Signal generation with universal models"""
        print("\n=== Testing Signal Generation with Universal Models ===")
        
        try:
            # Test 5.1: Signal generator initialization
            signal_generator = SignalGenerator()
            self.log_test("signal_generation", "initialization", "PASS", "SignalGenerator initialized")
            
            # Test 5.2: Universal model support
            if hasattr(signal_generator, 'is_universal_mode'):
                self.log_test("signal_generation", "universal_support", "PASS", "Universal mode support available")
                
                # Test 5.3: Universal model loading
                if hasattr(signal_generator, '_load_universal_models'):
                    self.log_test("signal_generation", "universal_loading", "PASS", "Universal model loading methods available")
                else:
                    self.log_test("signal_generation", "universal_loading", "FAIL", "Universal model loading methods missing")
                
                # Test 5.4: Universal prediction generation
                if hasattr(signal_generator, '_generate_universal_prediction'):
                    self.log_test("signal_generation", "universal_prediction", "PASS", "Universal prediction methods available")
                else:
                    self.log_test("signal_generation", "universal_prediction", "FAIL", "Universal prediction methods missing")
                    
            else:
                self.log_test("signal_generation", "universal_support", "FAIL", "Universal mode support missing")
                
        except Exception as e:
            self.log_test("signal_generation", "initialization", "FAIL", str(e))
    
    async def test_backward_compatibility(self):
        """Test 6: Backward compatibility with existing single-symbol training"""
        print("\n=== Testing Backward Compatibility ===")
        
        try:
            # Test 6.1: Single-symbol training endpoint
            test_symbol = "AAPL"
            training_config = {
                "epochs": 5,
                "batch_size": 32,
                "learning_rate": 0.001
            }
            
            response = requests.post(
                f"{self.base_url}/models/train/{test_symbol}",
                json=training_config,
                timeout=30
            )
            
            if response.status_code == 200:
                self.log_test("backward_compatibility", "single_symbol_training", "PASS", f"Single-symbol training for {test_symbol} works")
            else:
                self.log_test("backward_compatibility", "single_symbol_training", "FAIL", f"Status: {response.status_code}")
            
            # Test 6.2: Existing signal generation endpoints
            signal_response = requests.get(f"{self.base_url}/signals/{test_symbol}")
            if signal_response.status_code in [200, 404]:  # 404 is acceptable if no models trained yet
                self.log_test("backward_compatibility", "signal_endpoints", "PASS", "Signal endpoints accessible")
            else:
                self.log_test("backward_compatibility", "signal_endpoints", "FAIL", f"Status: {signal_response.status_code}")
            
            # Test 6.3: Model performance endpoints
            performance_response = requests.get(f"{self.base_url}/models/{test_symbol}/performance")
            if performance_response.status_code in [200, 404]:  # 404 is acceptable if no models exist
                self.log_test("backward_compatibility", "performance_endpoints", "PASS", "Performance endpoints accessible")
            else:
                self.log_test("backward_compatibility", "performance_endpoints", "FAIL", f"Status: {performance_response.status_code}")
                
        except Exception as e:
            self.log_test("backward_compatibility", "single_symbol_training", "FAIL", str(e))
    
    async def test_performance_validation(self):
        """Test 7: Performance validation against requirements"""
        print("\n=== Testing Performance Requirements ===")
        
        try:
            # Test 7.1: Training time estimation
            start_time = time.time()
            
            # Simulate a small universal training job for timing
            training_config = {
                "symbols": self.test_symbols[:2],  # Use 2 symbols for quick test
                "training_config": {
                    "base_model_epochs": 2,
                    "fine_tune_epochs": 1,
                    "ensemble_optimization": False,  # Skip for speed
                    "progressive_training": True
                },
                "data_config": {
                    "training_window_months": 1,  # Minimal data for speed
                    "validation_window_months": 1,
                    "feature_set": "reduced"
                }
            }
            
            response = requests.post(
                f"{self.base_url}/models/universal/train",
                json=training_config,
                timeout=60
            )
            
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data.get("job_id")
                
                # Monitor job completion
                max_wait_time = 300  # 5 minutes max
                check_interval = 10  # Check every 10 seconds
                elapsed_time = 0
                
                while elapsed_time < max_wait_time:
                    await asyncio.sleep(check_interval)
                    elapsed_time += check_interval
                    
                    status_response = requests.get(f"{self.base_url}/models/universal/train/status/{job_id}")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        job_status = status_data.get("status")
                        
                        if job_status in ["completed", "failed"]:
                            training_time = time.time() - start_time
                            
                            if job_status == "completed":
                                # Extrapolate to full universe (50 symbols)
                                estimated_full_time = (training_time / len(self.test_symbols[:2])) * 50
                                
                                if estimated_full_time < 18000:  # 5 hours = 18000 seconds
                                    self.log_test("performance_validation", "training_time", "PASS", 
                                                f"Estimated full training time: {estimated_full_time/3600:.1f} hours")
                                else:
                                    self.log_test("performance_validation", "training_time", "FAIL", 
                                                f"Estimated full training time: {estimated_full_time/3600:.1f} hours (exceeds 5h target)")
                            else:
                                self.log_test("performance_validation", "training_time", "FAIL", "Training job failed")
                            break
                    else:
                        self.log_test("performance_validation", "training_time", "FAIL", "Cannot monitor job status")
                        break
                else:
                    self.log_test("performance_validation", "training_time", "FAIL", "Training timeout (>5 minutes for test job)")
            else:
                self.log_test("performance_validation", "training_time", "FAIL", f"Cannot start training job: {response.status_code}")
            
            # Test 7.2: Memory usage validation
            import psutil
            process = psutil.Process()
            memory_usage_gb = process.memory_info().rss / (1024**3)
            
            if memory_usage_gb < 28:  # Within M4 32GB limit with buffer
                self.log_test("performance_validation", "memory_usage", "PASS", f"Memory usage: {memory_usage_gb:.1f}GB")
            else:
                self.log_test("performance_validation", "memory_usage", "FAIL", f"Memory usage: {memory_usage_gb:.1f}GB (exceeds 28GB limit)")
            
            # Test 7.3: API response time validation
            api_start = time.time()
            response = requests.get(f"{self.base_url}/models/universal/status")
            api_time = (time.time() - api_start) * 1000  # Convert to milliseconds
            
            if api_time < 100:  # Under 100ms requirement
                self.log_test("performance_validation", "api_response_time", "PASS", f"API response time: {api_time:.1f}ms")
            else:
                self.log_test("performance_validation", "api_response_time", "FAIL", f"API response time: {api_time:.1f}ms (exceeds 100ms limit)")
                
        except Exception as e:
            self.log_test("performance_validation", "training_time", "FAIL", str(e))
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("UNIVERSAL TRAINING SYSTEM - COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        total_time = time.time() - self.start_time
        print(f"Test Duration: {total_time:.1f} seconds")
        print(f"Test Timestamp: {datetime.now().isoformat()}")
        print(f"Test Symbols: {', '.join(self.test_symbols)}")
        
        # Calculate overall statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for category, tests in self.test_results.items():
            if category == "overall_status":
                continue
                
            print(f"\n--- {category.upper().replace('_', ' ')} ---")
            
            for test_name, result in tests.items():
                total_tests += 1
                status = result["status"]
                details = result["details"]
                
                if status == "PASS":
                    passed_tests += 1
                    print(f"  ✅ {test_name}: {details}")
                else:
                    failed_tests += 1
                    print(f"  ❌ {test_name}: {details}")
        
        # Overall assessment
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n--- OVERALL ASSESSMENT ---")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            overall_status = "READY FOR PRODUCTION"
            status_emoji = "🟢"
        elif success_rate >= 60:
            overall_status = "NEEDS MINOR FIXES"
            status_emoji = "🟡"
        else:
            overall_status = "NEEDS MAJOR FIXES"
            status_emoji = "🔴"
        
        self.test_results["overall_status"] = overall_status
        
        print(f"\n{status_emoji} SYSTEM STATUS: {overall_status}")
        
        # Recommendations
        print(f"\n--- RECOMMENDATIONS ---")
        if failed_tests == 0:
            print("  🎉 All tests passed! System is ready for production deployment.")
        else:
            print(f"  🔧 Address {failed_tests} failing test(s) before production deployment.")
            print("  📊 Review detailed test results above for specific issues.")
            print("  🔄 Re-run tests after fixes to validate improvements.")
        
        # Save detailed report to file
        report_file = f"/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/universal_system_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        print("="*80)
        
        return overall_status, success_rate
    
    async def run_all_tests(self):
        """Run all test suites"""
        print("Starting Universal Training System Comprehensive Test Suite...")
        print(f"Testing against: {self.base_url}")
        print(f"Test symbols: {', '.join(self.test_symbols)}")
        
        # Run all test suites
        await self.test_api_endpoints()
        await self.test_model_initialization()
        await self.test_mode_switching()
        await self.test_data_pipeline()
        await self.test_signal_generation()
        await self.test_backward_compatibility()
        await self.test_performance_validation()
        
        # Generate final report
        return self.generate_report()

async def main():
    """Main test execution function"""
    tester = UniversalSystemTester()
    
    try:
        overall_status, success_rate = await tester.run_all_tests()
        
        # Exit with appropriate code
        if success_rate >= 80:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except KeyboardInterrupt:
        print("\n⚠️  Test suite interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {str(e)}")
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())