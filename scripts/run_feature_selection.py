#!/usr/bin/env python3
"""
Feature Selection CLI Tool

This tool provides comprehensive feature selection analysis for the universal trading system.
It helps identify the most important features and reduces dimensionality from ~262 to 50-75 optimal features.

Usage:
    python scripts/run_feature_selection.py --mode analysis --symbols AAPL,MSFT,GOOGL
    python scripts/run_feature_selection.py --mode selection --method mutual_info --target-features 60
    python scripts/run_feature_selection.py --mode validate --model-path models/universal
"""

import argparse
import asyncio
import argparse
import logging
import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from data.data_pipeline import DataPipeline
from ml.feature_selector import UniversalFeatureSelector, FeatureSelectionConfig
from ml.universal_trainer import UniversalTrainer
from ml.universal_feature_engineering import UniversalFeatureEngineering
from trading.signal_generator import SignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('feature_selection.log')
    ]
)
logger = logging.getLogger(__name__)

class FeatureSelectionCLI:
    """Command-line interface for feature selection operations"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # Initialize components
        from database import DatabaseManager
        
        self.db_manager = DatabaseManager()
        self.data_pipeline = DataPipeline()
        self.feature_engineer = UniversalFeatureEngineering()
        self.feature_selector = UniversalFeatureSelector()
        
        self.results_dir = Path('feature_selection_results')
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("Feature selection CLI initialized")
    
    def _combine_feature_set(self, feature_set) -> pd.DataFrame:
        """Combine all feature DataFrames from FeatureSet into a single DataFrame"""
        try:
            combined_features = pd.DataFrame()
            
            # Combine all feature categories
            feature_dfs = [
                feature_set.technical_features,
                feature_set.market_microstructure,
                feature_set.sentiment_features,
                feature_set.macro_features,
                feature_set.cross_asset_features,
                feature_set.engineered_features
            ]
            
            for df in feature_dfs:
                if df is not None and not df.empty:
                    if combined_features.empty:
                        combined_features = df.copy()
                    else:
                        # Align indices and concatenate columns
                        combined_features = pd.concat([combined_features, df], axis=1, join='outer')
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error combining feature set: {e}")
            return pd.DataFrame()
    
    async def initialize_components(self):
        """Initialize database and feature selection components"""
        try:
            logger.info("All components already initialized in __init__")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def run_feature_analysis(self, symbols: List[str], days_back: int = 30) -> Dict:
        """Run comprehensive feature analysis"""
        logger.info(f"Starting feature analysis for symbols: {symbols}")
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Collect data for all symbols
            all_features = []
            all_targets = []
            
            for symbol in symbols:
                logger.info(f"Processing {symbol}...")
                
                # Get market data
                market_data = await self.data_pipeline.download_historical_data(
                    symbol, start_date, end_date
                )
                
                if market_data is None or market_data.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Engineer features
                feature_set = await self.feature_engineer.engineer_features(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    include_cross_asset=True,
                    training_mode=True
                )
                
                if feature_set is None:
                    logger.warning(f"No features generated for {symbol}")
                    continue
                
                # Combine all features into a single DataFrame
                features_df = self._combine_feature_set(feature_set)
                
                if features_df is None or features_df.empty:
                    logger.warning(f"No combined features available for {symbol}")
                    continue
                
                # Separate features and targets
                feature_cols = [col for col in features_df.columns if col != 'target']
                if 'target' in features_df.columns:
                    all_features.append(features_df[feature_cols])
                    all_targets.append(features_df['target'])
                    logger.info(f"Collected {len(feature_cols)} features for {symbol}")
                else:
                    logger.warning(f"No target column found for {symbol}")
            
            if not all_features:
                raise ValueError("No valid feature data collected")
            
            # Combine all data
            import pandas as pd
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_targets = pd.concat(all_targets, ignore_index=True)
            
            logger.info(f"Combined dataset: {combined_features.shape[0]} samples, {combined_features.shape[1]} features")
            
            # Run feature analysis
            analysis_results = await self.feature_selector.analyze_features(
                combined_features, combined_targets
            )
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = self.results_dir / f'feature_analysis_{timestamp}.json'
            
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"Analysis results saved to {results_file}")
            
            # Print summary
            self._print_analysis_summary(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Feature analysis failed: {e}")
            raise
    
    async def run_feature_selection(self, method: str, target_features: int, 
                                  symbols: List[str], days_back: int = 30) -> Dict:
        """Run feature selection with specified method"""
        logger.info(f"Starting feature selection: method={method}, target={target_features}")
        
        try:
            # First run analysis to get the data
            analysis_results = await self.run_feature_analysis(symbols, days_back)
            
            # Update config with selection parameters
            config = FeatureSelectionConfig(
                selection_method=method,
                target_feature_count=target_features
            )
            self.feature_selector.config = config
            
            # Get the combined dataset again (could be optimized)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            all_features = []
            all_targets = []
            
            for symbol in symbols:
                market_data = await self.data_pipeline.download_historical_data(
                    symbol, start_date, end_date
                )
                if market_data is None or market_data.empty:
                    continue
                
                feature_set = await self.feature_engineer.engineer_features(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    include_cross_asset=True,
                    training_mode=True
                )
                if feature_set is None:
                    logger.warning(f"No features generated for {symbol}")
                    continue
                
                # Combine all features into a single DataFrame
                features_df = self._combine_feature_set(feature_set)
                
                if features_df is None or features_df.empty:
                    logger.warning(f"No combined features available for {symbol}")
                    continue
                
                feature_cols = [col for col in features_df.columns if col != 'target']
                if 'target' in features_df.columns:
                    all_features.append(features_df[feature_cols])
                    all_targets.append(features_df['target'])
            
            if not all_features:
                raise ValueError("No valid feature data collected")
            
            import pandas as pd
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_targets = pd.concat(all_targets, ignore_index=True)
            
            # Run feature selection
            selection_results = await self.feature_selector.select_features(
                combined_features, combined_targets
            )
            
            # Save selection results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = self.results_dir / f'feature_selection_{method}_{target_features}_{timestamp}.json'
            
            with open(results_file, 'w') as f:
                json.dump(selection_results, f, indent=2, default=str)
            
            logger.info(f"Selection results saved to {results_file}")
            
            # Print summary
            self._print_selection_summary(selection_results)
            
            return selection_results
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            raise
    
    async def validate_selection(self, model_path: str, selection_results_file: Optional[str] = None) -> Dict:
        """Validate feature selection results against trained models"""
        logger.info(f"Validating feature selection with model path: {model_path}")
        
        try:
            # Load selection results
            if selection_results_file:
                with open(selection_results_file, 'r') as f:
                    selection_results = json.load(f)
            else:
                # Find the most recent selection results
                selection_files = list(self.results_dir.glob('feature_selection_*.json'))
                if not selection_files:
                    raise ValueError("No feature selection results found")
                
                latest_file = max(selection_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    selection_results = json.load(f)
                logger.info(f"Using selection results from {latest_file}")
            
            # Initialize signal generator for validation
            signal_generator = SignalGenerator()
            
            # Load models
            test_symbols = ['AAPL', 'MSFT', 'GOOGL']  # Test with common symbols
            await signal_generator.initialize_models(test_symbols)
            
            # Validate feature compatibility
            validation_results = {
                'model_path': model_path,
                'selection_file': selection_results_file,
                'selected_features': selection_results.get('selected_features', []),
                'feature_count': len(selection_results.get('selected_features', [])),
                'validation_timestamp': datetime.now().isoformat(),
                'compatibility_check': {},
                'performance_impact': {}
            }
            
            # Check if selected features are compatible with model expectations
            selected_features = selection_results.get('selected_features', [])
            
            for symbol in test_symbols:
                try:
                    # Test feature preparation with selected features
                    signal_generator.selected_features = selected_features
                    
                    # Get sample data for testing
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)
                    
                    market_data = await self.data_pipeline.download_historical_data(
                        symbol, start_date, end_date
                    )
                    
                    if market_data is not None and not market_data.empty:
                        # Test feature preparation
                        feature_set = await self.feature_engineer.engineer_features(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            include_cross_asset=True,
                            training_mode=True
                        )
                        
                        if feature_set is not None:
                            features_df = self._combine_feature_set(feature_set)
                            if features_df is not None:
                                available_features = [col for col in features_df.columns if col != 'target']
                                missing_features = set(selected_features) - set(available_features)
                                
                                validation_results['compatibility_check'][symbol] = {
                                    'total_available_features': len(available_features),
                                    'selected_features_available': len(selected_features) - len(missing_features),
                                    'missing_features': list(missing_features),
                                    'compatibility_score': (len(selected_features) - len(missing_features)) / len(selected_features) if selected_features else 0
                                }
                        
                except Exception as symbol_error:
                    validation_results['compatibility_check'][symbol] = {
                        'error': str(symbol_error)
                    }
            
            # Save validation results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = self.results_dir / f'validation_results_{timestamp}.json'
            
            with open(results_file, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            logger.info(f"Validation results saved to {results_file}")
            
            # Print summary
            self._print_validation_summary(validation_results)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def _print_analysis_summary(self, results: Dict):
        """Print feature analysis summary"""
        print("\n" + "="*60)
        print("FEATURE ANALYSIS SUMMARY")
        print("="*60)
        
        if 'feature_stats' in results:
            stats = results['feature_stats']
            print(f"Total Features Analyzed: {stats.get('total_features', 'N/A')}")
            print(f"Features with High Correlation: {stats.get('high_correlation_count', 'N/A')}")
            print(f"Features with Low Variance: {stats.get('low_variance_count', 'N/A')}")
        
        if 'importance_scores' in results:
            importance = results['importance_scores']
            print(f"\nTop 10 Most Important Features:")
            for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
                print(f"  {i:2d}. {feature}: {score:.4f}")
        
        print("\n" + "="*60)
    
    def _print_selection_summary(self, results: Dict):
        """Print feature selection summary"""
        print("\n" + "="*60)
        print("FEATURE SELECTION SUMMARY")
        print("="*60)
        
        selected_features = results.get('selected_features', [])
        print(f"Selected Features: {len(selected_features)}")
        print(f"Selection Method: {results.get('method', 'N/A')}")
        print(f"Selection Score: {results.get('selection_score', 'N/A')}")
        
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"\nPerformance Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        
        print(f"\nSelected Features:")
        for i, feature in enumerate(selected_features[:20], 1):  # Show first 20
            print(f"  {i:2d}. {feature}")
        
        if len(selected_features) > 20:
            print(f"  ... and {len(selected_features) - 20} more")
        
        print("\n" + "="*60)
    
    def _print_validation_summary(self, results: Dict):
        """Print validation summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        print(f"Selected Features: {results.get('feature_count', 'N/A')}")
        print(f"Model Path: {results.get('model_path', 'N/A')}")
        
        if 'compatibility_check' in results:
            print(f"\nCompatibility Check:")
            for symbol, check in results['compatibility_check'].items():
                if 'error' in check:
                    print(f"  {symbol}: ERROR - {check['error']}")
                else:
                    score = check.get('compatibility_score', 0)
                    print(f"  {symbol}: {score:.2%} compatible ({check.get('selected_features_available', 0)}/{results.get('feature_count', 0)} features)")
        
        print("\n" + "="*60)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Feature Selection CLI Tool for Universal Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run feature analysis
  python scripts/run_feature_selection.py --mode analysis --symbols AAPL,MSFT,GOOGL
  
  # Run feature selection with mutual information
  python scripts/run_feature_selection.py --mode selection --method mutual_info --target-features 60 --symbols AAPL,MSFT
  
  # Validate selection results
  python scripts/run_feature_selection.py --mode validate --model-path models/universal
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['analysis', 'selection', 'validate'],
        required=True,
        help='Operation mode: analysis, selection, or validate'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        default='AAPL,MSFT,GOOGL,NVDA,TSLA,AAPL,META,AMD,PLTR,AMZN,GOOGL,MSFT',
        help='Comma-separated list of symbols to analyze (default: AAPL,MSFT,GOOGL)'
    )
    
    parser.add_argument(
        '--method',
        choices=['mutual_info', 'correlation', 'variance', 'recursive', 'lasso'],
        default='mutual_info',
        help='Feature selection method (default: mutual_info)'
    )
    
    parser.add_argument(
        '--target-features',
        type=int,
        default=65,
        help='Target number of features to select (default: 65)'
    )
    
    parser.add_argument(
        '--days-back',
        type=int,
        default=30,
        help='Number of days of historical data to use (default: 30)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/universal',
        help='Path to model directory for validation (default: models/universal)'
    )
    
    parser.add_argument(
        '--selection-file',
        type=str,
        help='Path to specific selection results file for validation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

async def main():
    """Main CLI function"""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Initialize CLI
    cli = FeatureSelectionCLI()
    await cli.initialize_components()
    
    try:
        if args.mode == 'analysis':
            await cli.run_feature_analysis(symbols, args.days_back)
            
        elif args.mode == 'selection':
            await cli.run_feature_selection(
                args.method, args.target_features, symbols, args.days_back
            )
            
        elif args.mode == 'validate':
            await cli.validate_selection(args.model_path, args.selection_file)
        
        logger.info("Operation completed successfully")
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())