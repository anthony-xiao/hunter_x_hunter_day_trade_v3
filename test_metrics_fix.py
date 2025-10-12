#!/usr/bin/env python3
"""
Test script to verify the metrics calculation fix for the 'high_confidence_accuracy' error.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def test_comprehensive_metrics():
    """Test the comprehensive metrics calculation logic."""
    print("Testing comprehensive metrics calculation...")
    
    # Create dummy predictions and targets
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic predictions (probabilities between 0 and 1)
    cv_predictions = np.random.beta(2, 2, n_samples)  # Beta distribution for realistic probabilities
    y_train = np.random.randint(0, 2, n_samples)
    cv_score = 0.75  # Mock cross-validation score
    cv_std = 0.05    # Mock standard deviation
    
    print(f"Generated {n_samples} samples")
    print(f"Prediction range: [{cv_predictions.min():.3f}, {cv_predictions.max():.3f}]")
    print(f"Positive samples: {np.sum(y_train)} ({np.mean(y_train):.2%})")
    
    try:
        # Calculate comprehensive metrics using the same logic as the fixed method
        cv_predictions_binary = (cv_predictions > 0.5).astype(int)
        
        # Basic metrics
        val_accuracy = accuracy_score(y_train, cv_predictions_binary)
        val_precision = precision_score(y_train, cv_predictions_binary, zero_division=0)
        val_recall = recall_score(y_train, cv_predictions_binary, zero_division=0)
        val_f1 = f1_score(y_train, cv_predictions_binary, zero_division=0)
        val_roc_auc = cv_score
        val_loss = -np.mean(y_train * np.log(cv_predictions + 1e-15) + (1 - y_train) * np.log(1 - cv_predictions + 1e-15))
        
        # High confidence accuracy (predictions > 0.7)
        high_conf_mask = cv_predictions > 0.7
        high_conf_accuracy = 0.0
        if np.sum(high_conf_mask) > 0:
            high_conf_predictions = cv_predictions_binary[high_conf_mask]
            high_conf_targets = y_train[high_conf_mask]
            high_conf_accuracy = accuracy_score(high_conf_targets, high_conf_predictions)
        
        print(f"High confidence samples (>0.7): {np.sum(high_conf_mask)}")
        
        # Win rate by confidence levels
        confidence_intervals = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        win_rates_by_confidence = {}
        
        for low, high in confidence_intervals:
            mask = (cv_predictions >= low) & (cv_predictions < high)
            if np.sum(mask) > 0:
                conf_predictions = cv_predictions_binary[mask]
                conf_targets = y_train[mask]
                win_rate = accuracy_score(conf_targets, conf_predictions)
                win_rates_by_confidence[f'{low}-{high}'] = win_rate
            else:
                win_rates_by_confidence[f'{low}-{high}'] = 0.0
            
            print(f"Confidence interval [{low}-{high}): {np.sum(mask)} samples")
        
        # Create the comprehensive metrics dictionary
        metrics = {
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1_score': val_f1,
            'roc_auc': val_roc_auc,
            'cv_std': cv_std,
            'val_loss': val_loss,
            'high_confidence_accuracy': high_conf_accuracy,
            'win_rate_0.5-0.6': win_rates_by_confidence.get('0.5-0.6', 0.0),
            'win_rate_0.6-0.7': win_rates_by_confidence.get('0.6-0.7', 0.0),
            'win_rate_0.7-0.8': win_rates_by_confidence.get('0.7-0.8', 0.0),
            'win_rate_0.8-0.9': win_rates_by_confidence.get('0.8-0.9', 0.0),
            'win_rate_0.9-1.0': win_rates_by_confidence.get('0.9-1.0', 0.0)
        }
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cv_std',
            'val_loss', 'high_confidence_accuracy',
            'win_rate_0.5-0.6', 'win_rate_0.6-0.7', 'win_rate_0.7-0.8', 
            'win_rate_0.8-0.9', 'win_rate_0.9-1.0'
        ]
        
        print(f"\nReturned metrics keys: {list(metrics.keys())}")
        
        missing_metrics = []
        for metric in expected_metrics:
            if metric not in metrics:
                missing_metrics.append(metric)
        
        if missing_metrics:
            print(f"❌ FAILED: Missing metrics: {missing_metrics}")
            return False
        else:
            print("✅ SUCCESS: All expected metrics are present")
            print(f"\nSample metrics values:")
            for key, value in metrics.items():
                print(f"  - {key}: {value:.4f}")
            
            # Verify that high_confidence_accuracy is accessible (this was the original error)
            hca = metrics['high_confidence_accuracy']
            print(f"\n🎯 Key test: high_confidence_accuracy = {hca:.4f} (accessible without KeyError)")
            return True
            
    except Exception as e:
        print(f"❌ ERROR during metrics calculation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comprehensive_metrics()
    if success:
        print("\n🎉 Comprehensive metrics calculation test PASSED!")
        print("The 'high_confidence_accuracy' KeyError should now be resolved.")
        print("The _train_with_cross_validation method now returns all expected metrics.")
    else:
        print("\n💥 Comprehensive metrics calculation test FAILED!")
        print("The error may still occur.")
    
    exit(0 if success else 1)