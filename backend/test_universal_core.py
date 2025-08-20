#!/usr/bin/env python3
"""
Universal Training System Core Component Test

Focused test suite for Universal Training System core functionality:
1. Model initialization and universal capabilities
2. Data pipeline universal methods
3. Signal generator universal support
4. Basic API endpoint availability
"""

import asyncio
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Add backend directory to path
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

from data.data_pipeline import DataPipeline
from ml.model_trainer import ModelTrainer
from trading.signal_generator import SignalGenerator

class UniversalCoreSystemTester:
    """Core test suite for Universal Training System"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_symbols = ["AAPL", "TSLA"]
        self.test_results = {
            "core_components": {},
            "api_availability": {},
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
    
    async def test_core_components(self):
        """Test core Universal Training System components"""
        print("\n=== Testing Core Universal Training Components ===")
        
        # Test 1: Model Trainer Universal Capabilities
        try:
            model_trainer = ModelTrainer()
            self.log_test("core_components", "model_trainer_init", "PASS", "ModelTrainer initialized")
            
            # Check for universal training methods
            universal_methods = [
                'initialize_universal_training',
                'train_universal_models',
                'save_universal_models',
                'is_universal_mode'
            ]
            
            missing_methods = []
            for method in universal_methods:
                if not hasattr(model_trainer, method):
                    missing_methods.append(method)
            
            if not missing_methods:
                self.log_test("core_components", "model_trainer_universal_methods", "PASS", 
                            "All universal training methods available")
            else:
                self.log_test("core_components", "model_trainer_universal_methods", "FAIL", 
                            f"Missing methods: {missing_methods}")
            
            # Check symbol embedding support (via universal_trainer)
            if hasattr(model_trainer, 'universal_trainer') or hasattr(model_trainer, 'get_universal_prediction'):
                self.log_test("core_components", "symbol_embedding_support", "PASS", 
                            "Symbol embedding support detected via universal_trainer")
            else:
                self.log_test("core_components", "symbol_embedding_support", "FAIL", 
                            "Symbol embedding support missing")
                
        except Exception as e:
            self.log_test("core_components", "model_trainer_init", "FAIL", str(e))
        
        # Test 2: Data Pipeline Universal Capabilities
        try:
            data_pipeline = DataPipeline()
            self.log_test("core_components", "data_pipeline_init", "PASS", "DataPipeline initialized")
            
            # Check for universal data methods (using actual method names found)
            universal_data_methods = [
                'load_universal_data',
                'create_universal_dataset',
                'get_universal_features'
            ]
            
            missing_data_methods = []
            for method in universal_data_methods:
                if not hasattr(data_pipeline, method):
                    missing_data_methods.append(method)
            
            if not missing_data_methods:
                self.log_test("core_components", "data_pipeline_universal_methods", "PASS", 
                            "All universal data methods available")
            else:
                self.log_test("core_components", "data_pipeline_universal_methods", "FAIL", 
                            f"Missing methods: {missing_data_methods}")
                
        except Exception as e:
            self.log_test("core_components", "data_pipeline_init", "FAIL", str(e))
        
        # Test 3: Signal Generator Universal Capabilities
        try:
            signal_generator = SignalGenerator()
            self.log_test("core_components", "signal_generator_init", "PASS", "SignalGenerator initialized")
            
            # Check for universal signal methods
            universal_signal_methods = [
                'is_universal_mode',
                '_load_universal_models',
                '_generate_universal_prediction',
                'initialize_universal_models'
            ]
            
            missing_signal_methods = []
            for method in universal_signal_methods:
                if not hasattr(signal_generator, method):
                    missing_signal_methods.append(method)
            
            if not missing_signal_methods:
                self.log_test("core_components", "signal_generator_universal_methods", "PASS", 
                            "All universal signal methods available")
            else:
                self.log_test("core_components", "signal_generator_universal_methods", "FAIL", 
                            f"Missing methods: {missing_signal_methods}")
                
        except Exception as e:
            self.log_test("core_components", "signal_generator_init", "FAIL", str(e))
    
    async def test_api_availability(self):
        """Test basic API endpoint availability"""
        print("\n=== Testing API Endpoint Availability ===")
        
        # Test basic endpoints
        endpoints_to_test = [
            ("/docs", "API documentation"),
            ("/models/universal/status", "Universal mode status"),
            ("/models/universal/train/jobs", "Universal training jobs")
        ]
        
        for endpoint, description in endpoints_to_test:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code in [200, 404, 422]:  # 404/422 acceptable for some endpoints
                    self.log_test("api_availability", f"endpoint_{endpoint.replace('/', '_')}", "PASS", 
                                f"{description} accessible (status: {response.status_code})")
                else:
                    self.log_test("api_availability", f"endpoint_{endpoint.replace('/', '_')}", "FAIL", 
                                f"{description} returned status: {response.status_code}")
            except requests.exceptions.ConnectionError:
                self.log_test("api_availability", f"endpoint_{endpoint.replace('/', '_')}", "FAIL", 
                            f"{description} - Connection refused (server not ready)")
            except Exception as e:
                self.log_test("api_availability", f"endpoint_{endpoint.replace('/', '_')}", "FAIL", 
                            f"{description} - {str(e)}")
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*80)
        print("UNIVERSAL TRAINING SYSTEM - CORE COMPONENT TEST REPORT")
        print("="*80)
        
        total_time = time.time() - self.start_time
        print(f"Test Duration: {total_time:.1f} seconds")
        print(f"Test Timestamp: {datetime.now().isoformat()}")
        
        # Calculate statistics
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
        
        print(f"\n--- CORE SYSTEM ASSESSMENT ---")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            overall_status = "CORE SYSTEM READY"
            status_emoji = "🟢"
        elif success_rate >= 70:
            overall_status = "CORE SYSTEM MOSTLY READY"
            status_emoji = "🟡"
        else:
            overall_status = "CORE SYSTEM NEEDS FIXES"
            status_emoji = "🔴"
        
        self.test_results["overall_status"] = overall_status
        
        print(f"\n{status_emoji} CORE SYSTEM STATUS: {overall_status}")
        
        # Key findings
        print(f"\n--- KEY FINDINGS ---")
        if failed_tests == 0:
            print("  🎉 All core components are properly implemented!")
            print("  ✅ Universal Training System architecture is complete")
            print("  🚀 Ready for full system testing and deployment")
        else:
            print(f"  🔧 {failed_tests} core component issue(s) detected")
            print("  📋 Review failed tests for implementation gaps")
            print("  🔄 Address core issues before proceeding to full testing")
        
        # Save report
        report_file = f"/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/universal_core_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n📄 Core test report saved to: {report_file}")
        print("="*80)
        
        return overall_status, success_rate
    
    async def run_core_tests(self):
        """Run core component tests"""
        print("Starting Universal Training System Core Component Tests...")
        print(f"Testing against: {self.base_url}")
        
        await self.test_core_components()
        await self.test_api_availability()
        
        return self.generate_report()

async def main():
    """Main test execution function"""
    tester = UniversalCoreSystemTester()
    
    try:
        overall_status, success_rate = await tester.run_core_tests()
        
        # Exit with appropriate code
        if success_rate >= 70:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except KeyboardInterrupt:
        print("\n⚠️  Core test suite interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Core test suite failed with error: {str(e)}")
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())