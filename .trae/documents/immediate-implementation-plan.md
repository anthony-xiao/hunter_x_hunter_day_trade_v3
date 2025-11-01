# Immediate Implementation Plan (Stepwise)

## Scope
- Replace `random_forest` with `catboost` in the statistical ensemble.
- Use 18 months of training data with 3-month walk-forward validation windows.
- Add advanced intraday features: VWAP bands, anchored VWAP, ORB signals, and volatility-adjusted scaling.

## Changes Implemented
- Model architecture
  - Added `ModelType.CATBOOST` and integrated `CatBoostClassifier` creation in `backend/ml/universal_model_architectures.py`.
  - Replaced Random Forest with CatBoost in ensemble creation, training, and evaluation paths in `backend/ml/universal_trainer.py`.
  - Updated ensemble metadata and default weights to include `catboost`.
  - Added CatBoost hyperparameters to `UniversalTrainingConfig`.
- Training configuration
  - Confirmed `base_training_window=18` and `base_validation_window=3` in `UniversalTrainingConfig`.
  - Added `enable_walk_forward_validation=True` and `walk_forward_window_months=3` for clarity (used by planning; can be wired into scheduling next).
- Feature engineering
  - VWAP deviation bands: `vwap_band_upper_1/2`, `vwap_band_lower_1/2`.
  - Anchored VWAP: `anchored_vwap_session`, `anchored_vwap_prevday`.
  - VWAP momentum 5-min: `vwap_momentum_5`.
  - ORB features: `orb_high`, `orb_low`, `orb_range`, `orb_breakout_up/down`, `orb_range_expansion`.
  - Volatility scaling helper: `rv_scaled_5`.
- Dependencies
  - Added `catboost==1.2.5` to `backend/requirements.txt`.

## How To Run (Step-by-Step)
1) Install dependencies
   - `python -m pip install -r backend/requirements.txt`
2) Configure training job
   - Ensure data availability spans ≥18 months.
   - Set/confirm `UniversalTrainingConfig` in your training entrypoint (`backend/main.py`) uses defaults:
     - `base_training_window=18`, `base_validation_window=3`.
     - `enable_walk_forward_validation=True`, `walk_forward_window_months=3`.
3) Train models
   - Kick off universal training (Phase 3: statistical models):
     - LightGBM, XGBoost, CatBoost will train with 2D aggregated features.
     - Ensemble weights optimized via validation loss; default weights start `{'lightgbm':0.40,'xgboost':0.35,'catboost':0.25}`.
4) Validate minute-level performance
   - Monitor: win rate, average return (bp), profit factor, Sharpe, max drawdown.
   - Segment by time-of-day (06:00–07:00), by exit type (stop/limit), and sector (semiconductors, automotive).
5) Deploy to paper trading (limited scope)
   - Run only on symbols with consistent liquidity and positive validation expectancy.
   - Enforce risk limits (existing): stop loss/take profit; add volatility-aware sizing downstream as next step.

## Walk-Forward Validation Plan (3-Month Windows)
- Rolling windows (example): Train on months [t-18..t-3], validate on [t-3..t], advance by 1 month.
- Aggregate metrics across folds; use profit-based score to select best configurations.
- Track stability: variance of metrics by fold, threshold drift, feature importance consistency.

## Acceptance Criteria
- Ensemble trains successfully with CatBoost (no RandomForest paths in active ensemble).
- Intraday features computed without NaN surges or extreme values post-cleaning.
- Validation profit metrics improve vs prior baseline (targeting win rate ↑, average return > 0 bp).
- Inference latency remains under the 100ms budget per symbol batch.

## Rollback Plan
- If CatBoost underperforms, reduce its weight (`catboost` down to 0.10–0.15) while keeping it in ensemble for diversity.
- Feature flags can disable new features at load-time via feature selection pruning if required.

## Next Up (after this step)
- Wire volatility-adjusted sizing (ATR, realized vol) into `risk_manager.py` and `execution_engine.py`.
- Add time-of-day entry filters and dynamic stop (ATR×k) controls.
- Run per-symbol walk-forward backtests with the new features and CatBoost ensemble.
