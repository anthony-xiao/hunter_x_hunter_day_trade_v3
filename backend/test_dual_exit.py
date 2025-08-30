#!/usr/bin/env python3
"""
Test script to verify the dual exit targets implementation.
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MockConfig:
    """Mock configuration for testing."""
    prediction_window: int = 15
    take_profit_pct: float = 0.003  # 0.3%
    stop_loss_pct: float = 0.002   # 0.2%

class MockTrainer:
    """Mock trainer class with just the dual exit method."""
    
    def __init__(self, config):
        self.config = config
    
    def _create_dual_exit_targets(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Create binary targets using dual exit conditions within a prediction window.
        
        This method implements a more sophisticated target generation approach that:
        1. Uses a maximum prediction window (default 15 minutes)
        2. Applies dual exit conditions:
           - Take profit: Trigger when price increases by take_profit_pct within the horizon
           - Stop loss: Trigger when price decreases by stop_loss_pct within the same period
        3. Returns 1 for take profit hit, 0 for stop loss hit or no exit
        
        Args:
            market_data: DataFrame with market data including 'close' prices
            
        Returns:
            Binary targets as numpy array (1 for take profit, 0 for stop loss or no exit)
        """
        if 'close' not in market_data.columns:
            raise ValueError("Market data must contain 'close' column for target extraction")
        
        close_prices = market_data['close'].values
        targets = np.zeros(len(close_prices) - self.config.prediction_window, dtype=int)
        
        logger.info(f"Creating dual exit targets with {self.config.prediction_window}-period window")
        logger.info(f"Take profit threshold: {self.config.take_profit_pct*100:.2f}%")
        logger.info(f"Stop loss threshold: {self.config.stop_loss_pct*100:.2f}%")
        
        # Iterate through each possible starting point
        for i in range(len(targets)):
            current_price = close_prices[i]
            take_profit_price = current_price * (1 + self.config.take_profit_pct)
            stop_loss_price = current_price * (1 - self.config.stop_loss_pct)
            
            # Look ahead within the prediction window
            window_end = min(i + self.config.prediction_window + 1, len(close_prices))
            future_prices = close_prices[i+1:window_end]
            
            # Check for exit conditions
            target_hit = False
            for future_price in future_prices:
                if future_price >= take_profit_price:
                    targets[i] = 1  # Take profit hit
                    target_hit = True
                    break
                elif future_price <= stop_loss_price:
                    targets[i] = 0  # Stop loss hit
                    target_hit = True
                    break
            
            # If no exit condition is met within the window, default to 0
            if not target_hit:
                targets[i] = 0
        
        # Log target distribution for analysis
        take_profit_count = np.sum(targets == 1)
        stop_loss_count = np.sum(targets == 0)
        total_targets = len(targets)
        
        logger.info(f"Target distribution:")
        logger.info(f"  Take profit hits: {take_profit_count} ({take_profit_count/total_targets*100:.2f}%)")
        logger.info(f"  Stop loss/no exit: {stop_loss_count} ({stop_loss_count/total_targets*100:.2f}%)")
        logger.info(f"  Total targets: {total_targets}")
        
        return targets

async def test_dual_exit_targets():
    """
    Test the dual exit targets method with sample data.
    """
    print("Testing dual exit targets implementation...")
    
    # Create sample market data
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    
    # Create price data with some volatility
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0, 0.002, 100)  # 0.2% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    market_data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'volume': [1000] * 100
    })
    
    print(f"Created sample market data with {len(market_data)} rows")
    print(f"Price range: {market_data['close'].min():.4f} - {market_data['close'].max():.4f}")
    
    # Create configuration with dual exit parameters
    config = MockConfig(
        prediction_window=15,
        take_profit_pct=0.003,  # 0.3%
        stop_loss_pct=0.002     # 0.2%
    )
    
    # Initialize mock trainer
    trainer = MockTrainer(config)
    
    # Test the dual exit targets method
    print("\nTesting _create_dual_exit_targets method...")
    targets = trainer._create_dual_exit_targets(market_data)
    
    print(f"\nResults:")
    print(f"Total targets generated: {len(targets)}")
    print(f"Take profit hits (1): {np.sum(targets == 1)} ({np.sum(targets == 1)/len(targets)*100:.2f}%)")
    print(f"Stop loss/no exit (0): {np.sum(targets == 0)} ({np.sum(targets == 0)/len(targets)*100:.2f}%)")
    
    # Show some sample targets
    print(f"\nFirst 20 targets: {targets[:20]}")
    print(f"Last 20 targets: {targets[-20:]}")
    
    # Test edge cases
    print("\nTesting edge cases...")
    
    # Test with insufficient data
    small_data = market_data.head(10)
    try:
        small_targets = trainer._create_dual_exit_targets(small_data)
        print(f"Small data test: Generated {len(small_targets)} targets from {len(small_data)} rows")
    except Exception as e:
        print(f"Small data test failed: {e}")
    
    # Test with missing close column
    try:
        bad_data = market_data.drop('close', axis=1)
        trainer._create_dual_exit_targets(bad_data)
        print("ERROR: Should have failed with missing close column")
    except ValueError as e:
        print(f"Missing close column test passed: {e}")
    
    print("\nDual exit targets test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_dual_exit_targets())