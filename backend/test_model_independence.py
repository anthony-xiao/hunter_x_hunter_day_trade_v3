#!/usr/bin/env python3
"""
Direct test to verify that create_symbol_specific_head creates independent model instances.
This test focuses specifically on the model cloning fix without extensive data loading.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from ml.universal_model_architectures import UniversalModelArchitectures

async def test_model_independence():
    """
    Test that create_symbol_specific_head creates independent model instances
    and doesn't share trainable layer states between symbols.
    """
    print("\n=== Testing Model Independence Fix ===")
    
    try:
        # Step 1: Initialize components
        print("\n1. Initializing components...")
        
        # Step 2: Create a base model
        print("\n2. Creating base LSTM model...")
        architectures = UniversalModelArchitectures(
            num_symbols=2,              # Mock symbol count
            symbol_embedding_dim=32     # Mock embedding dimension
        )
        
        # Create base model with proper parameters
        config = {
            'lstm_units': 50,
            'dropout': 0.3,
            'l2_reg': 0.01,
            'learning_rate': 0.001
        }
        
        base_model = architectures.create_universal_lstm(
            sequence_length=30,
            feature_dim=100,
            config=config
        )
        print(f"Base model created with {len(base_model.layers)} layers")
        
        # Step 3: Create symbol-specific models for different symbols
        print("\n3. Creating symbol-specific models...")
        
        # Create two symbol-specific models
        config = {'layers_to_unfreeze': 2, 'dropout': 0.2}
        
        symbol_model_1 = architectures.create_symbol_specific_head(
            base_model=base_model,
            symbol_id=0,  # AAPL
            config=config
        )
        
        # Get trainable status of first model's layers
        trainable_status_1 = [layer.trainable for layer in symbol_model_1.layers]
        print(f"Symbol model 1 (AAPL) trainable layers: {sum(trainable_status_1)}/{len(trainable_status_1)}")
        
        symbol_model_2 = architectures.create_symbol_specific_head(
            base_model=base_model,
            symbol_id=1,  # TSLA
            config=config
        )
        
        # Get trainable status of second model's layers
        trainable_status_2 = [layer.trainable for layer in symbol_model_2.layers]
        print(f"Symbol model 2 (TSLA) trainable layers: {sum(trainable_status_2)}/{len(trainable_status_2)}")
        
        # Step 4: Verify independence
        print("\n4. Verifying model independence...")
        
        # Check if models are different instances
        models_are_different = symbol_model_1 is not symbol_model_2
        print(f"Models are different instances: {models_are_different}")
        
        # Check if they have the same layer configuration
        same_layer_count = len(symbol_model_1.layers) == len(symbol_model_2.layers)
        print(f"Models have same layer count: {same_layer_count}")
        
        # Check if trainable status is the same (should be for proper cloning)
        same_trainable_status = trainable_status_1 == trainable_status_2
        print(f"Models have same trainable configuration: {same_trainable_status}")
        
        # Step 5: Test that modifying one doesn't affect the other
        print("\n5. Testing independence by modifying one model...")
        
        # Modify trainable status of first model's last layer
        if len(symbol_model_1.layers) > 0:
            original_trainable = symbol_model_1.layers[-1].trainable
            symbol_model_1.layers[-1].trainable = not original_trainable
            
            # Check if second model was affected
            model_2_last_layer_trainable = symbol_model_2.layers[-1].trainable
            independence_verified = model_2_last_layer_trainable == original_trainable
            
            print(f"Model 1 last layer trainable changed to: {symbol_model_1.layers[-1].trainable}")
            print(f"Model 2 last layer trainable remains: {model_2_last_layer_trainable}")
            print(f"Independence verified: {independence_verified}")
            
            # Step 6: Summary
            print("\n=== Test Results ===")
            if models_are_different and same_layer_count and same_trainable_status and independence_verified:
                print("✅ SUCCESS: Model cloning fix is working correctly!")
                print("   - Models are independent instances")
                print("   - Models have proper layer configuration")
                print("   - Modifying one model doesn't affect the other")
                return True
            else:
                print("❌ FAILURE: Model cloning fix has issues:")
                if not models_are_different:
                    print("   - Models are the same instance (not cloned)")
                if not same_layer_count:
                    print("   - Models have different layer counts")
                if not same_trainable_status:
                    print("   - Models have different trainable configurations")
                if not independence_verified:
                    print("   - Models are not independent (shared state)")
                return False
        else:
            print("❌ No layers found in models to test")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_model_independence())
    if result:
        print("\n🎉 Model independence test PASSED!")
    else:
        print("\n💥 Model independence test FAILED!")
        sys.exit(1)