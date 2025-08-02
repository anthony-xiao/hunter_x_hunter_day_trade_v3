#!/usr/bin/env python3
"""
Polygon Timestamp Test Script

This script tests what timestamps Polygon actually sends us and compares
them with what we're storing in the database.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
import requests
from typing import List, Dict, Any

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings


class PolygonTimestampTester:
    def __init__(self):
        self.api_key = settings.polygon_api_key
        self.base_url = "https://api.polygon.io"
        
    def test_polygon_aggregates(self, symbol: str = "AAPL", date: str = None):
        """
        Test what timestamps Polygon sends for minute aggregates
        """
        if date is None:
            # Use yesterday for a complete trading day
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            date = yesterday.strftime("%Y-%m-%d")
        
        print(f"\n=== POLYGON AGGREGATES TEST for {symbol} on {date} ===")
        
        # Polygon aggregates endpoint
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/minute/{date}/{date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,
            'apikey': self.api_key
        }
        
        try:
            print(f"Fetching from: {url}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' not in data or not data['results']:
                print(f"No data returned for {date}")
                return
            
            results = data['results']
            print(f"\nReceived {len(results)} minute bars")
            
            # Analyze first and last few timestamps
            print("\nFirst 5 timestamps from Polygon:")
            for i, bar in enumerate(results[:5]):
                timestamp_ms = bar['t']
                timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                timestamp_et = timestamp_dt.astimezone(timezone(timedelta(hours=-5)))  # EST
                
                print(f"  {i+1}. Raw: {timestamp_ms}")
                print(f"     UTC: {timestamp_dt}")
                print(f"     ET:  {timestamp_et}")
                print(f"     OHLCV: O:{bar['o']} H:{bar['h']} L:{bar['l']} C:{bar['c']} V:{bar['v']}")
                print()
            
            print("\nLast 5 timestamps from Polygon:")
            for i, bar in enumerate(results[-5:]):
                timestamp_ms = bar['t']
                timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                timestamp_et = timestamp_dt.astimezone(timezone(timedelta(hours=-5)))  # EST
                
                print(f"  {len(results)-4+i}. Raw: {timestamp_ms}")
                print(f"     UTC: {timestamp_dt}")
                print(f"     ET:  {timestamp_et}")
                print(f"     OHLCV: O:{bar['o']} H:{bar['h']} L:{bar['l']} C:{bar['c']} V:{bar['v']}")
                print()
            
            # Analyze hour distribution
            hour_counts = {}
            for bar in results:
                timestamp_dt = datetime.fromtimestamp(bar['t'] / 1000, tz=timezone.utc)
                timestamp_et = timestamp_dt.astimezone(timezone(timedelta(hours=-5)))  # EST
                hour_et = timestamp_et.hour
                hour_counts[hour_et] = hour_counts.get(hour_et, 0) + 1
            
            print("\nHour distribution (ET):")
            for hour in sorted(hour_counts.keys()):
                print(f"  Hour {hour:02d}: {hour_counts[hour]} bars")
            
            # Check if we have expected trading hours
            print("\nTrading hours analysis:")
            pre_market = sum(hour_counts.get(h, 0) for h in range(4, 10))  # 4 AM - 9:30 AM
            regular = sum(hour_counts.get(h, 0) for h in range(9, 16))      # 9:30 AM - 4 PM
            after_hours = sum(hour_counts.get(h, 0) for h in range(16, 21)) # 4 PM - 8 PM
            
            print(f"  Pre-market (4-9 AM ET): {pre_market} bars")
            print(f"  Regular (9-4 PM ET): {regular} bars")
            print(f"  After-hours (4-8 PM ET): {after_hours} bars")
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Polygon: {e}")
        except Exception as e:
            print(f"Error processing Polygon data: {e}")
    
    def test_timezone_conversion(self):
        """
        Test timezone conversion scenarios
        """
        print("\n=== TIMEZONE CONVERSION TEST ===")
        
        # Test timestamp: 2025-07-29 14:30:00 UTC (9:30 AM ET - market open)
        test_timestamp_utc = datetime(2025, 7, 29, 14, 30, 0, tzinfo=timezone.utc)
        
        print(f"Test timestamp (market open): {test_timestamp_utc}")
        
        # Convert to different timezones
        et_tz = timezone(timedelta(hours=-5))  # EST
        shanghai_tz = timezone(timedelta(hours=8))  # Asia/Shanghai
        
        timestamp_et = test_timestamp_utc.astimezone(et_tz)
        timestamp_shanghai = test_timestamp_utc.astimezone(shanghai_tz)
        
        print(f"  In ET: {timestamp_et}")
        print(f"  In Shanghai: {timestamp_shanghai}")
        
        # Test what happens if we store Shanghai time as UTC
        fake_utc = timestamp_shanghai.replace(tzinfo=timezone.utc)
        print(f"  If Shanghai time stored as UTC: {fake_utc}")
        
        # Test timestamp: 2025-07-29 21:00:00 UTC (4:00 PM ET - market close)
        test_timestamp_close = datetime(2025, 7, 29, 21, 0, 0, tzinfo=timezone.utc)
        
        print(f"\nTest timestamp (market close): {test_timestamp_close}")
        
        timestamp_close_et = test_timestamp_close.astimezone(et_tz)
        timestamp_close_shanghai = test_timestamp_close.astimezone(shanghai_tz)
        
        print(f"  In ET: {timestamp_close_et}")
        print(f"  In Shanghai: {timestamp_close_shanghai}")
        
        # This explains the 16:00 pattern!
        fake_utc_close = timestamp_close_shanghai.replace(tzinfo=timezone.utc)
        print(f"  If Shanghai time stored as UTC: {fake_utc_close}")
        
        print("\n=== DIAGNOSIS ===")
        print("The 16:00 pattern suggests:")
        print("1. Polygon sends correct UTC timestamps")
        print("2. Our database timezone is Asia/Shanghai (+8)")
        print("3. When we store UTC timestamps, PostgreSQL interprets them as Shanghai time")
        print("4. This shifts everything by +8 hours")
        print("5. Market close (4 PM ET = 21:00 UTC) becomes 05:00 Shanghai next day")
        print("6. But when displayed, it shows as the local part: 21:00 -> 21:00 Shanghai")
        print("7. The 16:00 you see is likely 4 PM ET data incorrectly timezone-shifted")


def main():
    tester = PolygonTimestampTester()
    
    # Test what Polygon actually sends
    tester.test_polygon_aggregates("AAPL", "2025-07-29")
    
    # Test timezone conversion scenarios
    tester.test_timezone_conversion()
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Set PostgreSQL timezone to UTC: SET timezone = 'UTC';")
    print("2. Or ensure all timestamps are explicitly stored with timezone info")
    print("3. Re-import all market data with correct timezone handling")
    print("4. Update data pipeline to handle timezone conversions properly")


if __name__ == "__main__":
    main()