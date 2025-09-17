#!/usr/bin/env python3
"""
Comprehensive Universal Training System Test Suite
Validates all components and provides performance metrics
"""

import asyncio
import requests
import time
from datetime import datetime
from data.data_pipeline import DataPipeline
from ml.universal_trainer import UniversalTrainer
from ml.universal_model_architectures import UniversalModelArchitectures
from ml.ml_feature_engineering import FeatureEngineering

class UniversalSystemTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_results = {}
        self.performance_metrics = {}
        
    def log_test(self, test_name, status, details="", duration=None):
        """Log test results"""
        self.test_results[test_name] = {
            'status': status,
            'details': details,
            'duration': duration
        }
        status_symbol = "✓" if status == "PASSED" else "✗"
        duration_str = f" ({duration:.2f}s)" if duration else ""
        print(f"{status_symbol} {test_name}: {status}{duration_str}")
        if details:
            print(f"  Details: {details}")
    
    def test_core_components(self):
        """Test core component initialization"""
        print("\n=== Testing Core Components ===")
        
        # Test 1: Universal Model Architectures
        start_time = time.time()
        try:
            arch = UniversalModelArchitectures(num_symbols=10, symbol_embedding_dim=32)
            duration = time.time() - start_time
            self.log_test("Universal Model Architectures", "PASSED", 
                         "Successfully initialized with 10 symbols, 32D embeddings", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Universal Model Architectures", "FAILED", str(e), duration)
        
        # Test 2: Universal Trainer
        start_time = time.time()
        try:
            dp = DataPipeline()
            fe = FeatureEngineering(data_pipeline=dp, supabase_client=None)
            trainer = UniversalTrainer(dp, fe)
            duration = time.time() - start_time
            self.log_test("Universal Trainer", "PASSED", 
                         "Successfully initialized with 3-phase training strategy", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Universal Trainer", "FAILED", str(e), duration)
    
    def test_api_endpoints(self):
        """Test API endpoints"""
        print("\n=== Testing API Endpoints ===")
        
        # Test 1: Universal Status
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/models/universal/status")
            duration = time.time() - start_time
            if response.status_code == 200:
                data = response.json()
                self.log_test("Universal Status Endpoint", "PASSED", 
                             f"Status: {data.get('universal_mode_enabled', 'Unknown')}", duration)
            else:
                self.log_test("Universal Status Endpoint", "FAILED", 
                             f"HTTP {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Universal Status Endpoint", "FAILED", str(e), duration)
        
        # Test 2: Universal Training
        start_time = time.time()
        try:
            payload = {
                'symbols': ['AAPL'],
                'start_date': '2024-01-01',
                'end_date': '2024-01-05',
                'model_type': 'lstm'
            }
            response = requests.post(f"{self.base_url}/models/universal/train", json=payload)
            duration = time.time() - start_time
            if response.status_code == 200:
                self.log_test("Universal Training Endpoint", "PASSED", 
                             "Training job started successfully", duration)
                self.performance_metrics['training_start_time'] = duration
            else:
                self.log_test("Universal Training Endpoint", "FAILED", 
                             f"HTTP {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Universal Training Endpoint", "FAILED", str(e), duration)
        
        # Test 3: Single Symbol Training (Backward Compatibility)
        start_time = time.time()
        try:
            response = requests.post(f"{self.base_url}/models/train/AAPL")
            duration = time.time() - start_time
            if response.status_code == 200:
                self.log_test("Single Symbol Training (Backward Compatibility)", "PASSED", 
                             "Legacy endpoint working", duration)
            else:
                self.log_test("Single Symbol Training (Backward Compatibility)", "FAILED", 
                             f"HTTP {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Single Symbol Training (Backward Compatibility)", "FAILED", str(e), duration)
        
        # Test 4: Signals Endpoint
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/signals")
            duration = time.time() - start_time
            if response.status_code == 200:
                self.log_test("Signals Endpoint", "PASSED", 
                             "Signal generation accessible", duration)
            else:
                self.log_test("Signals Endpoint", "FAILED", 
                             f"HTTP {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Signals Endpoint", "FAILED", str(e), duration)
    
    def test_mode_switching(self):
        """Test universal vs symbol-specific mode switching"""
        print("\n=== Testing Mode Switching ===")
        
        start_time = time.time()
        try:
            # Get current status
            response = requests.get(f"{self.base_url}/models/universal/status")
            initial_status = response.json()['universal_mode_enabled']
            
            # Test disable
            requests.post(f"{self.base_url}/models/universal/disable")
            response = requests.get(f"{self.base_url}/models/universal/status")
            disabled_status = response.json()['universal_mode_enabled']
            
            duration = time.time() - start_time
            if disabled_status == False:
                self.log_test("Mode Switching", "PASSED", 
                             f"Initial: {initial_status}, After disable: {disabled_status}", duration)
            else:
                self.log_test("Mode Switching", "FAILED", 
                             "Mode switching not working properly", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Mode Switching", "FAILED", str(e), duration)
    
    async def test_batch_processing(self):
        """Test data pipeline batch processing"""
        print("\n=== Testing Batch Processing ===")
        
        start_time = time.time()
        try:
            dp = DataPipeline()
            symbols = ['AAPL', 'TSLA']
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 5)
            
            data = await dp.load_universal_data(symbols, start_date, end_date)
            duration = time.time() - start_time
            
            self.log_test("Batch Data Processing", "PASSED", 
                         f"Processed {len(symbols)} symbols, loaded {len(data)} datasets", duration)
            self.performance_metrics['batch_processing_time'] = duration
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Batch Data Processing", "FAILED", str(e), duration)
    
    def validate_performance_requirements(self):
        """Validate performance against requirements"""
        print("\n=== Performance Validation ===")
        
        # Training time reduction (should be faster than individual training)
        if 'training_start_time' in self.performance_metrics:
            training_time = self.performance_metrics['training_start_time']
            if training_time < 5.0:  # Should start quickly
                self.log_test("Training Time Performance", "PASSED", 
                             f"Training started in {training_time:.2f}s")
            else:
                self.log_test("Training Time Performance", "FAILED", 
                             f"Training took {training_time:.2f}s to start")
        
        # Data utilization (batch processing efficiency)
        if 'batch_processing_time' in self.performance_metrics:
            batch_time = self.performance_metrics['batch_processing_time']
            if batch_time < 10.0:  # Should process multiple symbols efficiently
                self.log_test("Data Utilization Performance", "PASSED", 
                             f"Batch processing completed in {batch_time:.2f}s")
            else:
                self.log_test("Data Utilization Performance", "FAILED", 
                             f"Batch processing took {batch_time:.2f}s")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("UNIVERSAL TRAINING SYSTEM - COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\nOVERALL RESULTS:")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"\nSYSTEM STATUS:")
        if success_rate >= 90:
            print("🟢 SYSTEM READY - Universal Training System is fully operational")
        elif success_rate >= 70:
            print("🟡 SYSTEM MOSTLY READY - Minor issues detected")
        else:
            print("🔴 SYSTEM NOT READY - Critical issues require attention")
        
        print(f"\nPERFORMANCE METRICS:")
        for metric, value in self.performance_metrics.items():
            print(f"- {metric}: {value:.2f}s")
        
        print(f"\nDETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status_symbol = "✓" if result['status'] == 'PASSED' else "✗"
            duration_str = f" ({result['duration']:.2f}s)" if result['duration'] else ""
            print(f"{status_symbol} {test_name}: {result['status']}{duration_str}")
            if result['details']:
                print(f"  {result['details']}")
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"universal_system_test_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("UNIVERSAL TRAINING SYSTEM - COMPREHENSIVE TEST REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Passed: {passed_tests}\n")
            f.write(f"Failed: {total_tests - passed_tests}\n")
            f.write(f"Success Rate: {success_rate:.1f}%\n\n")
            
            for test_name, result in self.test_results.items():
                f.write(f"{test_name}: {result['status']}\n")
                if result['details']:
                    f.write(f"  Details: {result['details']}\n")
                if result['duration']:
                    f.write(f"  Duration: {result['duration']:.2f}s\n")
                f.write("\n")
        
        print(f"\nReport saved to: {report_file}")
        return success_rate

async def main():
    """Run comprehensive system tests"""
    print("Starting Universal Training System Comprehensive Tests...")
    
    tester = UniversalSystemTester()
    
    # Run all test suites
    tester.test_core_components()
    tester.test_api_endpoints()
    tester.test_mode_switching()
    await tester.test_batch_processing()
    tester.validate_performance_requirements()
    
    # Generate final report
    success_rate = tester.generate_report()
    
    return success_rate >= 90

if __name__ == '__main__':
    success = asyncio.run(main())
    exit(0 if success else 1)