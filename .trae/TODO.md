# TODO:

- [x] load_training_metadata: Load correct feature count (150) from training metadata instead of using hardcoded defaults (priority: High)
- [x] update_default_configs: Update _set_default_model_configs to use 150 features instead of 153 (priority: High)
- [x] fix_feature_selection: Modify _prepare_features to select exactly 150 features from the 169 available (priority: High)
- [x] add_confidence_logging: Add detailed logging to show individual model predictions and confidence calculation breakdown (priority: High)
- [x] improve_individual_confidence: Add variability to individual model confidence calculations using model-specific factors (priority: High)
- [x] improve_ensemble_confidence: Improve ensemble confidence calculation to be less dependent on standard deviation (priority: High)
- [x] update_model_creation: Update model creation methods to use training metadata feature count (priority: Medium)
- [x] test_fix: Test the fix by running the trading start command (priority: Medium)
- [x] test_confidence_fix: Test the confidence calculation improvements and verify variability (priority: Medium)
- [ ] fix_featureset_len_error: Fix bootstrap error where len() is called on FeatureSet object - replace with proper validation and fix iterrows() call (**IN PROGRESS**) (priority: High)
