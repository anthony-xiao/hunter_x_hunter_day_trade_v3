#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3')
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

from loguru import logger
from data.data_pipeline import DataPipeline
from ml.universal_feature_engineering import UniversalFeatureEngineering
from ml.universal_trainer import UniversalTrainer

# Configure loguru to output DEBUG level to stdout
logger.remove()
logger.add(sys.stdout, level="DEBUG")

logger.info("Starting feature validation debug test")

# Initialize components
data_pipeline = DataPipeline()
feature_engineering = UniversalFeatureEngineering()
trainer = UniversalTrainer(data_pipeline, feature_engineering)

# Load data for a single symbol
symbols = ['AAPL']
start_date = '2024-01-01'
end_date = '2024-01-31'

logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
universal_data = data_pipeline.load_universal_data(symbols, start_date, end_date)

logger.info("Engineering features")
universal_features = feature_engineering.engineer_universal_features(universal_data)

logger.info("Preparing training data")
X, y = feature_engineering.prepare_universal_training_data(universal_features)

logger.info(f"Final feature matrix shape: {X.shape}")
logger.info(f"Feature columns: {list(X.columns)}")

# Now run the validation manually
logger.info("Running manual feature validation...")
feature_engineering._validate_feature_dimensions(X, expected_total=178)

logger.info("Feature validation debug complete")