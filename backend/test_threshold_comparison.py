#!/usr/bin/env python3
"""
Comprehensive test to compare old vs new threshold values and demonstrate improved balance.
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
class TestConfig:
    """Configuration for testing different threshold scenarios."""
    prediction_window: int = 30
    take_profit_pct: float = 0.005
    stop_loss_pct: float = 0.003

class ThresholdTester:
    """Tester class to compare different threshold configurations."""
    
    def __init__(self, config):
        self.config = config
    
    def _create_dual_exit_targets(self, market_data: pd.DataFrame) -> np.ndarray:
        """Create binary targets using dual exit conditions."""
        if 'close' not in market_data.columns:
            raise ValueError("Market data must contain 'close' column for target extraction")
        
        close_prices = market_data['close'].values
        targets = np.zeros(len(close_prices) - self.config.prediction_window, dtype=int)
        
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
        
        return targets
    
    def analyze_targets(self, targets: np.ndarray, scenario_name: str) -> dict:
        """Analyze target distribution and return statistics."""
        take_profit_count = np.sum(targets == 1)
        stop_loss_count = np.sum(targets == 0)
        total_targets = len(targets)
        
        take_profit_pct = (take_profit_count / total_targets * 100) if total_targets > 0 else 0
        stop_loss_pct = (stop_loss_count / total_targets * 100) if total_targets > 0 else 0
        
        stats = {
            'scenario': scenario_name,
            'take_profit_threshold': f"{self.config.take_profit_pct*100:.2f}%",
            'stop_loss_threshold': f"{self.config.stop_loss_pct*100:.2f}%",
            'total_targets': total_targets,
            'take_profit_count': take_profit_count,
            'stop_loss_count': stop_loss_count,
            'take_profit_pct': take_profit_pct,
            'stop_loss_pct': stop_loss_pct,
            'balance_ratio': take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else float('inf')
        }
        
        return stats

def create_market_scenarios():
    """Create different market scenarios for testing."""
    scenarios = {}
    
    # Scenario 1: Normal volatility market
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1min')
    base_price = 100.0
    returns = np.random.normal(0, 0.002, 500)  # 0.2% volatility
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    scenarios['normal_volatility'] = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'volume': [1000] * 500
    })
    
    # Scenario 2: High volatility market
    np.random.seed(123)
    returns_high = np.random.normal(0, 0.005, 500)  # 0.5% volatility
    prices_high = [base_price]
    for ret in returns_high[1:]:
        prices_high.append(prices_high[-1] * (1 + ret))
    
    scenarios['high_volatility'] = pd.DataFrame({
        'timestamp': dates,
        'close': prices_high,
        'open': prices_high,
        'high': [p * 1.001 for p in prices_high],
        'low': [p * 0.999 for p in prices_high],
        'volume': [1000] * 500
    })
    
    # Scenario 3: Trending market (upward bias)
    np.random.seed(456)
    returns_trend = np.random.normal(0.0005, 0.003, 500)  # Slight upward bias with 0.3% volatility
    prices_trend = [base_price]
    for ret in returns_trend[1:]:
        prices_trend.append(prices_trend[-1] * (1 + ret))
    
    scenarios['trending_market'] = pd.DataFrame({
        'timestamp': dates,
        'close': prices_trend,
        'open': prices_trend,
        'high': [p * 1.001 for p in prices_trend],
        'low': [p * 0.999 for p in prices_trend],
        'volume': [1000] * 500
    })
    
    return scenarios

async def test_threshold_comparison():
    """Compare old vs new threshold configurations across different market scenarios."""
    print("=" * 80)
    print("THRESHOLD COMPARISON TEST")
    print("=" * 80)
    
    # Create market scenarios
    scenarios = create_market_scenarios()
    
    # Define old and new configurations
    old_config = TestConfig(
        prediction_window=30,
        take_profit_pct=0.005,  # 0.5%
        stop_loss_pct=0.003     # 0.3%
    )
    
    new_config = TestConfig(
        prediction_window=30,
        take_profit_pct=0.003,  # 0.3%
        stop_loss_pct=0.002     # 0.2%
    )
    
    # Test both configurations
    old_tester = ThresholdTester(old_config)
    new_tester = ThresholdTester(new_config)
    
    all_results = []
    
    for scenario_name, market_data in scenarios.items():
        print(f"\n📊 Testing Scenario: {scenario_name.upper()}")
        print(f"Market data: {len(market_data)} rows, Price range: {market_data['close'].min():.4f} - {market_data['close'].max():.4f}")
        
        # Test old configuration
        old_targets = old_tester._create_dual_exit_targets(market_data)
        old_stats = old_tester.analyze_targets(old_targets, f"{scenario_name}_old")
        
        # Test new configuration
        new_targets = new_tester._create_dual_exit_targets(market_data)
        new_stats = new_tester.analyze_targets(new_targets, f"{scenario_name}_new")
        
        all_results.extend([old_stats, new_stats])
        
        # Display comparison
        print(f"\n  OLD THRESHOLDS (TP: {old_config.take_profit_pct*100:.1f}%, SL: {old_config.stop_loss_pct*100:.1f}%):")
        print(f"    Take Profit: {old_stats['take_profit_count']:3d} ({old_stats['take_profit_pct']:5.1f}%)")
        print(f"    Stop Loss:   {old_stats['stop_loss_count']:3d} ({old_stats['stop_loss_pct']:5.1f}%)")
        print(f"    Balance Ratio: {old_stats['balance_ratio']:.3f}")
        
        print(f"\n  NEW THRESHOLDS (TP: {new_config.take_profit_pct*100:.1f}%, SL: {new_config.stop_loss_pct*100:.1f}%):")
        print(f"    Take Profit: {new_stats['take_profit_count']:3d} ({new_stats['take_profit_pct']:5.1f}%)")
        print(f"    Stop Loss:   {new_stats['stop_loss_count']:3d} ({new_stats['stop_loss_pct']:5.1f}%)")
        print(f"    Balance Ratio: {new_stats['balance_ratio']:.3f}")
        
        # Calculate improvement
        balance_improvement = new_stats['balance_ratio'] - old_stats['balance_ratio']
        print(f"\n  📈 IMPROVEMENT: Balance ratio improved by {balance_improvement:+.3f}")
        if balance_improvement > 0:
            print(f"     ✅ Better balance achieved with new thresholds!")
        else:
            print(f"     ⚠️  Balance decreased with new thresholds")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    old_results = [r for r in all_results if r['scenario'].endswith('_old')]
    new_results = [r for r in all_results if r['scenario'].endswith('_new')]
    
    avg_old_balance = np.mean([r['balance_ratio'] for r in old_results])
    avg_new_balance = np.mean([r['balance_ratio'] for r in new_results])
    
    avg_old_tp_pct = np.mean([r['take_profit_pct'] for r in old_results])
    avg_new_tp_pct = np.mean([r['take_profit_pct'] for r in new_results])
    
    print(f"\nAverage across all scenarios:")
    print(f"  OLD: Take Profit {avg_old_tp_pct:.1f}%, Balance Ratio: {avg_old_balance:.3f}")
    print(f"  NEW: Take Profit {avg_new_tp_pct:.1f}%, Balance Ratio: {avg_new_balance:.3f}")
    print(f"\n🎯 OVERALL IMPROVEMENT: {avg_new_balance - avg_old_balance:+.3f}")
    
    if avg_new_balance > avg_old_balance:
        print(f"✅ SUCCESS: New thresholds provide better target balance!")
    else:
        print(f"❌ The new thresholds did not improve balance")
    
    print(f"\n💡 CONCLUSION:")
    print(f"   Lowering both take profit (0.5% → 0.3%) and stop loss (0.3% → 0.2%) thresholds")
    print(f"   resulted in {'improved' if avg_new_balance > avg_old_balance else 'decreased'} target balance across different market conditions.")
    print(f"   This should lead to {'better' if avg_new_balance > avg_old_balance else 'worse'} model training performance.")

if __name__ == "__main__":
    asyncio.run(test_threshold_comparison())