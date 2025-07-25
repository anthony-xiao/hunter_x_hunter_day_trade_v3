# Model Versioning System

This document describes the model versioning system implemented for the Hunter x Hunter Day Trade project.

## Overview

The model versioning system automatically creates timestamped directories for each training run, preventing model overwrites and allowing you to track and compare different model versions.

## Directory Structure

```
backend/models/
├── 20250724_150837/          # Timestamped model version
│   ├── lstm_model.h5         # LSTM neural network model
│   ├── transformer_model.h5  # Transformer neural network model
│   ├── random_forest_model.pkl # Random Forest model
│   ├── xgboost_model.pkl     # XGBoost model
│   ├── scalers.pkl           # Feature scalers
│   └── training_metadata.json # Training metadata and metrics
├── 20250724_150841/          # Another model version
│   └── ...
└── latest -> 20250724_150841 # Symlink to most recent version
```

## Features

### Automatic Versioning
- Each training run creates a new timestamped directory (format: `YYYYMMDD_HHMMSS`)
- Models are saved in their respective version directories
- A `latest` symlink always points to the most recent version

### Metadata Tracking
Each version includes a `training_metadata.json` file containing:
- Training timestamp
- Model configurations (hyperparameters, architecture)
- Performance metrics (accuracy, Sharpe ratio, win rate, max drawdown)
- Ensemble weights
- List of trained models

### Model Types Supported
- **Neural Networks**: LSTM, CNN, Transformer (saved as `.h5` files)
- **Traditional ML**: Random Forest, XGBoost (saved as `.pkl` files)
- **Scalers**: Feature preprocessing scalers (saved as `scalers.pkl`)

## Usage

### Training Models
When you run the training process:
```bash
python run.py
```

Models will automatically be saved to a new timestamped directory in `backend/models/`.

### Managing Model Versions

Use the `manage_models.py` utility for version management:

#### List All Versions
```bash
python manage_models.py list
```
Shows all available versions with creation time, model count, size, and best Sharpe ratio.

#### Show Version Details
```bash
python manage_models.py show 20250724_150837
```
Displays detailed information about a specific version including performance metrics and ensemble weights.

#### Load Specific Version
```bash
python manage_models.py load latest
python manage_models.py load 20250724_150837
```
Loads models from a specific version or the latest version.

#### Delete Old Version
```bash
python manage_models.py delete 20250724_150634
```
Deletes a specific model version (with confirmation prompt).

#### Cleanup Old Versions
```bash
python manage_models.py cleanup --keep 5
```
Keeps only the 5 most recent versions and deletes older ones.

## Code Integration

### ModelTrainer Class Methods

The `ModelTrainer` class now includes these version management methods:

- `list_model_versions()`: Returns list of all available versions with metadata
- `get_current_version()`: Returns the currently loaded version
- `delete_model_version(version)`: Deletes a specific version directory
- `load_models(version='latest')`: Loads models from specific version

### Loading Models in Code

```python
from ml.model_trainer import ModelTrainer

# Load latest models
trainer = ModelTrainer()
await trainer.load_models('latest')

# Load specific version
await trainer.load_models('20250724_150837')

# Check current version
current_version = trainer.get_current_version()
print(f"Currently using models from: {current_version}")
```

## Benefits

1. **No Data Loss**: Previous models are never overwritten
2. **Easy Comparison**: Compare performance across different training runs
3. **Rollback Capability**: Easily revert to previous model versions
4. **Metadata Tracking**: Comprehensive information about each training run
5. **Storage Management**: Tools to clean up old versions when needed
6. **Automated Workflow**: Versioning happens automatically during training

## Storage Considerations

- Each model version typically uses 2-3 MB of storage
- Use the cleanup utility regularly to manage disk space
- Consider keeping only the best-performing versions for production use

## Troubleshooting

### Missing 'latest' Symlink
If the `latest` symlink is missing, the system will automatically find the most recent timestamped directory.

### Corrupted Metadata
If `training_metadata.json` is corrupted, the system will still load models but won't display performance metrics.

### Permission Issues
Ensure the `backend/models/` directory has write permissions for creating new version directories.