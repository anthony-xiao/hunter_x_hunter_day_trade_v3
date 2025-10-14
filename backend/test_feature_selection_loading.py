#!/usr/bin/env python3
"""
Test script to verify if SignalGenerator is loading selected_feature_columns correctly
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from trading.signal_generator import SignalGenerator
from data.data_pipeline import DataPipeline

async def test_feature_selection_loading():
    """Test if SignalGenerator loads selected_feature_columns correctly"""
    
    print("=== Testing SignalGenerator Feature Selection Loading ===")
    
    # 1. Check if universal_metadata.json exists and has selected_feature_columns
    metadata_path = backend_dir / 'models' / 'universal' / 'universal_metadata.json'
    print(f"1. Checking metadata file: {metadata_path}")
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if 'feature_selection' in metadata and 'selected_feature_columns' in metadata['feature_selection']:
            selected_features = metadata['feature_selection']['selected_feature_columns']
            print(f"   ✓ Found {len(selected_features)} selected_feature_columns in metadata")
            print(f"   ✓ First 5 features: {selected_features[:5]}")
        else:
            print("   ✗ No selected_feature_columns found in metadata")
            return
    else:
        print(f"   ✗ Metadata file not found: {metadata_path}")
        return
    
    # 2. Initialize SignalGenerator and check if it loads the features
    print("\n2. Initializing SignalGenerator...")
    
    try:
        data_pipeline = DataPipeline()
        signal_generator = SignalGenerator(data_pipeline)
        
        # Check if selected_feature_columns was loaded
        if hasattr(signal_generator, 'selected_feature_columns') and signal_generator.selected_feature_columns:
            print(f"   ✓ SignalGenerator loaded {len(signal_generator.selected_feature_columns)} selected_feature_columns")
            print(f"   ✓ First 5 loaded features: {signal_generator.selected_feature_columns[:5]}")
            
            # Compare with metadata
            if signal_generator.selected_feature_columns == selected_features:
                print("   ✓ Loaded features match metadata exactly")
            else:
                print("   ⚠️  Loaded features differ from metadata")
                print(f"      Metadata count: {len(selected_features)}")
                print(f"      Loaded count: {len(signal_generator.selected_feature_columns)}")
        else:
            print("   ✗ SignalGenerator did not load selected_feature_columns")
            print(f"      selected_feature_columns attribute: {getattr(signal_generator, 'selected_feature_columns', 'NOT_FOUND')}")
        
        # Check other feature selection attributes
        print(f"\n3. Other feature selection attributes:")
        print(f"   selected_features: {getattr(signal_generator, 'selected_features', 'NOT_FOUND')}")
        print(f"   feature_selection_metadata: {getattr(signal_generator, 'feature_selection_metadata', 'NOT_FOUND')}")
        
        # Initialize models to trigger feature selection loading
        print(f"\n4. Initializing models to trigger feature selection loading...")
        symbols = ['AAPL', 'TSLA', 'NVDA']  # Test with a few symbols
        success = await signal_generator.initialize_models(symbols)
        
        if success:
            print("   ✓ Models initialized successfully")
            
            # Check again after initialization
            if hasattr(signal_generator, 'selected_feature_columns') and signal_generator.selected_feature_columns:
                print(f"   ✓ After initialization: {len(signal_generator.selected_feature_columns)} selected_feature_columns")
            else:
                print("   ✗ After initialization: still no selected_feature_columns")
        else:
            print("   ✗ Model initialization failed")
            
    except Exception as e:
        print(f"   ✗ Error initializing SignalGenerator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_feature_selection_loading())