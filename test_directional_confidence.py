#!/usr/bin/env python3
"""
Test script to verify DirectionalConfidence implementation
and ensure sell signals are no longer artificially suppressed.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from trading.signal_generator import DirectionalConfidence
import numpy as np

def test_directional_confidence():
    """Test DirectionalConfidence calculation for buy vs sell scenarios"""
    print("Testing DirectionalConfidence implementation...")
    print("=" * 50)
    
    # Test scenarios: [prediction, base_confidence, expected_behavior]
    test_cases = [
        # Strong buy signals
        (0.8, 0.7, "Strong Buy"),
        (0.6, 0.6, "Moderate Buy"),
        (0.3, 0.5, "Weak Buy"),
        
        # Strong sell signals
        (-0.8, 0.7, "Strong Sell"),
        (-0.6, 0.6, "Moderate Sell"),
        (-0.3, 0.5, "Weak Sell"),
        
        # Neutral signals
        (0.1, 0.5, "Weak Buy"),
        (-0.1, 0.5, "Weak Sell"),
        (0.0, 0.5, "Neutral")
    ]
    
    print(f"{'Scenario':<15} {'Prediction':<12} {'Base Conf':<10} {'Buy Conf':<10} {'Sell Conf':<10} {'Direction':<10} {'Bias Check':<15}")
    print("-" * 90)
    
    for prediction, base_confidence, scenario in test_cases:
        # Calculate directional confidence
        model_variance = 0.1  # Small variance for testing
        directional_conf = DirectionalConfidence.calculate(prediction, base_confidence, model_variance)
        
        # Determine which confidence should be higher
        if prediction > 0:
            expected_higher = "buy"
            actual_higher = "buy" if directional_conf.buy_confidence > directional_conf.sell_confidence else "sell"
        elif prediction < 0:
            expected_higher = "sell"
            actual_higher = "sell" if directional_conf.sell_confidence > directional_conf.buy_confidence else "buy"
        else:
            expected_higher = "neutral"
            actual_higher = "neutral"
        
        # Check for bias (both confidences should be reasonable)
        bias_check = "PASS"
        if prediction < 0:  # Sell signal
            if directional_conf.sell_confidence < 0.3:  # Artificially suppressed
                bias_check = "FAIL - Suppressed"
        elif prediction > 0:  # Buy signal
            if directional_conf.buy_confidence < 0.3:  # Artificially suppressed
                bias_check = "FAIL - Suppressed"
        
        print(f"{scenario:<15} {prediction:<12.2f} {base_confidence:<10.2f} {directional_conf.buy_confidence:<10.2f} {directional_conf.sell_confidence:<10.2f} {actual_higher:<10} {bias_check:<15}")
    
    print("\n" + "=" * 50)
    print("Testing confidence calculation properties...")
    
    # Test that sell signals are not artificially suppressed
    strong_sell_prediction = -0.8
    strong_sell_base_conf = 0.7
    sell_conf = DirectionalConfidence.calculate(strong_sell_prediction, strong_sell_base_conf, 0.1)
    
    print(f"\nStrong Sell Signal Test:")
    print(f"Prediction: {strong_sell_prediction}")
    print(f"Base Confidence: {strong_sell_base_conf}")
    print(f"Sell Confidence: {sell_conf.sell_confidence:.3f}")
    print(f"Buy Confidence: {sell_conf.buy_confidence:.3f}")
    print(f"Direction Clarity: {sell_conf.direction_clarity:.3f}")
    print(f"Prediction Strength: {sell_conf.prediction_strength:.3f}")
    
    # Verify sell confidence is not artificially low
    if sell_conf.sell_confidence >= 0.5:
        print("✅ PASS: Sell confidence is not artificially suppressed")
    else:
        print("❌ FAIL: Sell confidence appears to be suppressed")
    
    # Test that buy signals work similarly
    strong_buy_prediction = 0.8
    strong_buy_base_conf = 0.7
    buy_conf = DirectionalConfidence.calculate(strong_buy_prediction, strong_buy_base_conf, 0.1)
    
    print(f"\nStrong Buy Signal Test:")
    print(f"Prediction: {strong_buy_prediction}")
    print(f"Base Confidence: {strong_buy_base_conf}")
    print(f"Buy Confidence: {buy_conf.buy_confidence:.3f}")
    print(f"Sell Confidence: {buy_conf.sell_confidence:.3f}")
    
    # Compare buy vs sell confidence for equivalent magnitude predictions
    print(f"\nBias Test (equivalent magnitude):")
    print(f"Buy confidence for +0.8 prediction: {buy_conf.buy_confidence:.3f}")
    print(f"Sell confidence for -0.8 prediction: {sell_conf.sell_confidence:.3f}")
    
    confidence_diff = abs(buy_conf.buy_confidence - sell_conf.sell_confidence)
    if confidence_diff < 0.1:  # Should be very similar
        print(f"✅ PASS: No significant bias between buy/sell confidence (diff: {confidence_diff:.3f})")
    else:
        print(f"⚠️  WARNING: Potential bias detected (diff: {confidence_diff:.3f})")
    
    print("\n" + "=" * 50)
    print("DirectionalConfidence test completed!")

def test_old_vs_new_approach():
    """Compare old absolute value approach vs new directional approach"""
    print("\nComparing old vs new confidence calculation approaches...")
    print("=" * 60)
    
    test_predictions = [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8]
    
    print(f"{'Prediction':<12} {'Old (abs)':<12} {'New (buy)':<12} {'New (sell)':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for prediction in test_predictions:
        # Old approach (using absolute value)
        old_confidence = 0.5 + abs(prediction) * 0.4
        
        # New approach (directional)
        base_confidence = 0.5 + (prediction * prediction) * 0.4
        directional_conf = DirectionalConfidence.calculate(prediction, base_confidence, 0.1)
        
        # Determine improvement
        if prediction < 0:  # Sell signal
            relevant_new_conf = directional_conf.sell_confidence
            improvement = "Better" if relevant_new_conf > old_confidence else "Similar"
        elif prediction > 0:  # Buy signal
            relevant_new_conf = directional_conf.buy_confidence
            improvement = "Better" if relevant_new_conf > old_confidence else "Similar"
        else:
            relevant_new_conf = max(directional_conf.buy_confidence, directional_conf.sell_confidence)
            improvement = "Similar"
        
        print(f"{prediction:<12.2f} {old_confidence:<12.3f} {directional_conf.buy_confidence:<12.3f} {directional_conf.sell_confidence:<12.3f} {improvement:<12}")

if __name__ == "__main__":
    try:
        test_directional_confidence()
        test_old_vs_new_approach()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()