#!/usr/bin/env python3
"""
Test script to validate the bracket order fix.
This script tests the position validation logic that prevents bracket order errors.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from trading.execution_engine import ExecutionEngine, TradeSignal
from trading.execution_engine import PositionSizing

class MockTradingClient:
    """Mock trading client for testing"""
    def __init__(self):
        self.orders = []
        self.should_fail_bracket = False
    
    def submit_order(self, request):
        """Mock order submission"""
        # Simulate the Alpaca API error for bracket orders with existing positions
        if hasattr(request, 'order_class') and request.order_class and self.should_fail_bracket:
            raise Exception("bracket orders must be entry orders")
        
        # Create mock order response
        class MockOrder:
            def __init__(self):
                self.id = "test_order_123"
                self.symbol = request.symbol
                self.qty = request.qty
                self.side = request.side
                self.status = "new"
        
        order = MockOrder()
        self.orders.append(order)
        return order
    
    def get_asset(self, symbol):
        """Mock asset info"""
        class MockAsset:
            def __init__(self):
                self.shortable = True
        return MockAsset()

async def test_bracket_order_fix():
    """Test the bracket order fix logic"""
    print("=== Testing Bracket Order Fix ===")
    
    # Create execution engine with mock client
    execution_engine = ExecutionEngine()
    execution_engine.trading_client = MockTradingClient()
    execution_engine.is_trading = True
    
    # Test signal
    test_signal = TradeSignal(
        symbol="TSLA",
        action="buy",
        confidence=0.8,
        timestamp=datetime.now(timezone.utc),
        stop_loss=None,
        take_profit=None,
        predicted_return=0.02,
        risk_score=0.3
    )
    
    # Test sizing
    test_sizing = PositionSizing(
        base_size=1000.0,
        volatility_adjusted=950.0,
        confidence_adjusted=900.0,
        risk_adjusted=850.0,
        final_size=100.0,
        max_allowed=2000.0,
        reasoning="Test sizing for bracket order validation"
    )
    
    print("\n1. Testing buy signal with NO existing position (should use bracket order):")
    try:
        # Clear positions to simulate no existing position
        execution_engine.positions = {}
        
        order = await execution_engine._execute_buy_signal(test_signal, test_sizing)
        if order:
            print(f"✅ SUCCESS: Order placed successfully - {order.symbol}")
            print(f"   Order ID: {order.id}")
        else:
            print("❌ FAILED: No order returned")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    print("\n2. Testing buy signal WITH existing position (should use regular order):")
    try:
        # Add existing position to simulate existing position
        execution_engine.positions = {"TSLA": {"qty": 5, "side": "long"}}
        
        order = await execution_engine._execute_buy_signal(test_signal, test_sizing)
        if order:
            print(f"✅ SUCCESS: Order placed successfully - {order.symbol}")
            print(f"   Order ID: {order.id}")
        else:
            print("❌ FAILED: No order returned")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    print("\n3. Testing sell signal with NO existing position (should be rejected):")
    try:
        # Clear positions
        execution_engine.positions = {}
        test_signal.action = "sell"
        
        order = await execution_engine._execute_sell_signal(test_signal, test_sizing)
        if order is None:
            print(f"✅ SUCCESS: Sell signal correctly rejected - no existing position")
        else:
            print(f"❌ FAILED: Order should not have been placed - {order.symbol}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    print("\n4. Testing sell signal WITH existing LONG position (should close position):")
    try:
        # Add existing LONG position (positive quantity)
        class MockPosition:
            def __init__(self, qty):
                self.quantity = qty
        
        execution_engine.positions = {"TSLA": MockPosition(5)}
        
        order = await execution_engine._execute_sell_signal(test_signal, test_sizing)
        if order:
            print(f"✅ SUCCESS: Order placed successfully to close long position - {order.symbol}")
            print(f"   Order ID: {order.id}")
        else:
            print("❌ FAILED: No order returned")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        
    print("\n5. Testing sell signal WITH existing SHORT position (should be rejected):")
    try:
        # Add existing SHORT position (negative quantity)
        execution_engine.positions = {"TSLA": MockPosition(-5)}
        
        order = await execution_engine._execute_sell_signal(test_signal, test_sizing)
        if order is None:
            print(f"✅ SUCCESS: Sell signal correctly rejected - cannot sell short position")
        else:
            print(f"❌ FAILED: Order should not have been placed for short position")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    print("\n6. Simulating the original error scenario:")
    try:
        # Set up mock to fail bracket orders (simulating the original error)
        execution_engine.trading_client.should_fail_bracket = True
        execution_engine.positions = {"TSLA": {"qty": 5, "side": "long"}}  # Existing position
        test_signal.action = "buy"
        
        order = await execution_engine._execute_buy_signal(test_signal, test_sizing)
        if order:
            print(f"✅ SUCCESS: Fix working! Regular order used instead of bracket order")
            print(f"   Order ID: {order.id}")
        else:
            print("❌ FAILED: No order returned")
    except Exception as e:
        print(f"❌ FAILED: Error still occurs - {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_bracket_order_fix())