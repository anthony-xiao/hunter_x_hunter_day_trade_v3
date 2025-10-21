# Optimal LightGBM Configuration for Minute-to-Minute Day Trading

## 📊 YOUR SYSTEM SPECIFICATIONS

- **Training Data**: 12 months (~98,280 minute-level samples)
- **Validation Data**: 3 months (~24,570 minute-level samples)
- **Lookback Window**: 30 minutes
- **Trading Frequency**: Minute-to-minute
- **Goal**: Maximum profit from day trading

---

## 🎯 OPTIMAL LIGHTGBM CONFIGURATION

### **Complete Implementation Code**

```python
# ADD TO universal_model_architectures.py

import lightgbm as lgb
from typing import Dict, Optional
import numpy as np

def create_universal_lightgbm(
    self, 
    feature_dim: int, 
    config: Optional[Dict] = None,
    model_name: str = "universal_lightgbm"
) -> lgb.LGBMClassifier:
    """
    Create optimized LightGBM for minute-to-minute day trading
    
    Optimized for:
    - 12 months training data (~98K samples)
    - 3 months validation data (~24K samples)
    - 30-minute lookback window
    - Minute-level trading frequency
    - Maximum profit optimization
    """
    
    logger.info(f"Creating LightGBM model with {feature_dim} features for day trading")
    
    # OPTIMAL CONFIGURATION for your exact use case
    default_params = {
        # ==== CORE PARAMETERS ====
        'boosting_type': 'gbdt',              # Standard gradient boosting (most stable)
        'objective': 'binary',                # Buy vs No-Buy classification
        'metric': ['binary_logloss', 'auc'],  # Monitor both calibration and ranking
        
        # ==== TREE STRUCTURE (CRITICAL FOR 98K SAMPLES) ====
        'num_leaves': 63,                     # 2^6-1 = optimal for your data size
        'max_depth': 6,                       # Prevents overfitting on minute noise
        'min_data_in_leaf': 50,               # ~0.05% of data, ensures stable leaves
        'min_sum_hessian_in_leaf': 1e-3,      # Controls leaf weight (imbalance handling)
        
        # ==== LEARNING RATE (LOWER FOR NOISY MINUTE DATA) ====
        'learning_rate': 0.05,                # Conservative rate for better generalization
        'n_estimators': 500,                  # With early stopping, typically stops at 200-350
        
        # ==== SAMPLING (ANTI-OVERFITTING) ====
        'bagging_fraction': 0.8,              # Use 80% of data per iteration
        'bagging_freq': 5,                    # Resample every 5 iterations
        'feature_fraction': 0.8,              # Use 80% of features per tree
        
        # ==== REGULARIZATION (CRITICAL FOR MINUTE DATA) ====
        'lambda_l1': 0.5,                     # L1 removes weak features
        'lambda_l2': 0.5,                     # L2 smooths leaf weights
        'min_gain_to_split': 0.01,            # Minimum info gain to split
        'path_smooth': 0.1,                   # Smooths prediction paths (reduces volatility)
        
        # ==== CLASS IMBALANCE (TRADING SIGNALS) ====
        'is_unbalance': True,                 # Auto-handle imbalanced buy signals
        
        # ==== PERFORMANCE (SPEED OPTIMIZATION) ====
        'num_threads': -1,                    # Use all CPU cores
        'device': 'cpu',                      # CPU optimal for this data size
        'verbose': -1,                        # Suppress output
        
        # ==== ADVANCED ====
        'max_bin': 255,                       # Optimal binning (default)
        'categorical_feature': 'auto',        # Auto-detect categorical features
        
        # ==== REPRODUCIBILITY ====
        'random_state': 42,
        'deterministic': True
    }
    
    # Override with any provided config
    if config:
        default_params.update(config)
    
    # Create model
    model = lgb.LGBMClassifier(**default_params)
    
    logger.info("✅ LightGBM configured for minute-to-minute trading:")
    logger.info(f"   Trees: {default_params['n_estimators']} @ lr={default_params['learning_rate']}")
    logger.info(f"   Structure: depth={default_params['max_depth']}, leaves={default_params['num_leaves']}")
    logger.info(f"   Regularization: L1={default_params['lambda_l1']}, L2={default_params['lambda_l2']}")
    
    return model


def train_lightgbm_with_early_stopping(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Optional[Dict] = None
) -> lgb.LGBMClassifier:
    """
    Train LightGBM with early stopping and validation monitoring
    
    CRITICAL for preventing overfitting on minute-level data
    """
    
    logger.info("🚀 Training LightGBM with early stopping...")
    
    # Create model
    model = self.create_universal_lightgbm(X_train.shape[1], config)
    
    # Train with early stopping (CRITICAL)
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        eval_names=['validation'],
        eval_metric=['binary_logloss', 'auc'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Get results
    best_iteration = model.best_iteration_
    best_score = model.best_score_
    
    logger.info(f"✅ Training complete:")
    logger.info(f"   Best iteration: {best_iteration}/{model.n_estimators}")
    logger.info(f"   Best validation AUC: {best_score['validation']['auc']:.4f}")
    logger.info(f"   Best validation logloss: {best_score['validation']['binary_logloss']:.4f}")
    
    # Feature importance (helps identify what matters)
    feature_importance = model.feature_importances_
    top_10_indices = np.argsort(feature_importance)[-10:][::-1]
    
    logger.info(f"   Top 10 most important features:")
    for idx in top_10_indices:
        logger.info(f"      Feature {idx}: {feature_importance[idx]:.0f}")
    
    return model
```

---

## 📋 PARAMETER EXPLANATIONS

### **Tree Structure (Most Important)**

| Parameter | Value | Why This Value? |
|-----------|-------|-----------------|
| `num_leaves` | 63 | 2^6-1 = optimal for 98K samples. More = overfitting, Less = underfitting |
| `max_depth` | 6 | Matches num_leaves, prevents deep trees on noisy minute data |
| `min_data_in_leaf` | 50 | ~0.05% of data, ensures each leaf is statistically significant |

### **Learning & Regularization**

| Parameter | Value | Why This Value? |
|-----------|-------|-----------------|
| `learning_rate` | 0.05 | Lower than default (0.1) for noisy minute data |
| `n_estimators` | 500 | With early stopping, typically stops at 200-350 |
| `lambda_l1` | 0.5 | L1 removes weak features (many in trading data) |
| `lambda_l2` | 0.5 | L2 smooths leaf weights (prevents overfitting) |

### **Sampling (Anti-Overfitting)**

| Parameter | Value | Why This Value? |
|-----------|-------|-----------------|
| `bagging_fraction` | 0.8 | Use 80% of data per iteration (adds randomness) |
| `feature_fraction` | 0.8 | Use 80% of features per tree (prevents dominance) |
| `bagging_freq` | 5 | Resample every 5 iterations for diversity |

---

## ✅ IMPLEMENTATION CHECKLIST

### **Setup (10 minutes)**
- [ ] Install LightGBM: `pip install lightgbm`
- [ ] Add `create_universal_lightgbm()` to `universal_model_architectures.py`
- [ ] Add `train_lightgbm_with_early_stopping()` function
- [ ] Import lightgbm: `import lightgbm as lgb`

### **Integration (30 minutes)**
- [ ] Update ensemble to use [LightGBM, XGBoost, RandomForest]
- [ ] Set ensemble weights: LightGBM=40%, XGBoost=35%, RF=25%
- [ ] Remove SVM code and scaling requirements
- [ ] Update training pipeline to call new LightGBM function

---
