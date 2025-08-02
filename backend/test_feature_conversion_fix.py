#!/usr/bin/env python3
"""
Test script to verify the _convert_cached_features_to_dataframe fix
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add the backend directory to the Python path
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

from trading.signal_generator import SignalGenerator

async def test_feature_conversion_fix():
    """Test the enhanced _convert_cached_features_to_dataframe method"""
    
    # Initialize signal generator
    signal_generator = SignalGenerator()
    
    # Create test data with mixed scenarios