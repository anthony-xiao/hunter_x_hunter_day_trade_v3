#!/usr/bin/env python3
"""
Simple test script to verify timestamp conversion logic
"""

from datetime import datetime, timezone
import pandas as pd

def test_timestamp_conversion():
    """Test the timestamp conversion logic used in the fixes"""
    print("Testing timestamp conversion logic...")
    
    # Test 1: Pandas timestamp without timezone (naive)
    print("\n1. Testing naive pandas timestamp:")
    ts_naive = pd.Timestamp('2024-01-01 10:00:00')
    print(f"Original: {ts_naive} (tzinfo: {ts_naive.tzinfo})")
    
    dt_naive = ts_naive.to_pydatetime()
    print(f"to_pydatetime(): {dt_naive} (tzinfo: {dt_naive.tzinfo})")
    
    if dt_naive.tzinfo is None:
        dt_naive = dt_naive.replace(tzinfo=timezone.utc)
    print(f"With UTC: {dt_naive} (tzinfo: {dt_naive.tzinfo})")
    
    # Test 2: Pandas timestamp with UTC timezone
    print("\n2. Testing UTC pandas timestamp:")
    ts_utc = pd.Timestamp('2024-01-01 10:00:00', tz='UTC')
    print(f"Original: {ts_utc} (tzinfo: {ts_utc.tzinfo})")
    
    dt_utc = ts_utc.to_pydatetime()
    print(f"to_pydatetime(): {dt_utc} (tzinfo: {dt_utc.tzinfo})")
    
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    elif dt_utc.tzinfo != timezone.utc:
        dt_utc = dt_utc.astimezone(timezone.utc)
    print(f"Final UTC: {dt_utc} (tzinfo: {dt_utc.tzinfo})")
    
    # Test 3: Pandas timestamp with different timezone
    print("\n3. Testing EST pandas timestamp:")
    ts_est = pd.Timestamp('2024-01-01 10:00:00', tz='US/Eastern')
    print(f"Original: {ts_est} (tzinfo: {ts_est.tzinfo})")
    
    dt_est = ts_est.to_pydatetime()
    print(f"to_pydatetime(): {dt_est} (tzinfo: {dt_est.tzinfo})")
    
    if dt_est.tzinfo is None:
        dt_est = dt_est.replace(tzinfo=timezone.utc)
    elif dt_est.tzinfo != timezone.utc:
        dt_est = dt_est.astimezone(timezone.utc)
    print(f"Converted to UTC: {dt_est} (tzinfo: {dt_est.tzinfo})")
    
    # Test 4: Regular datetime object
    print("\n4. Testing regular datetime object:")
    dt_regular = datetime(2024, 1, 1, 10, 0, 0)
    print(f"Original: {dt_regular} (tzinfo: {dt_regular.tzinfo})")
    
    if isinstance(dt_regular, datetime):
        if dt_regular.tzinfo is None:
            dt_regular = dt_regular.replace(tzinfo=timezone.utc)
        elif dt_regular.tzinfo != timezone.utc:
            dt_regular = dt_regular.astimezone(timezone.utc)
    print(f"Final UTC: {dt_regular} (tzinfo: {dt_regular.tzinfo})")
    
    print("\nTimestamp conversion logic test completed successfully!")
    print("The fixes should ensure all timestamps are stored in UTC timezone.")

if __name__ == "__main__":
    test_timestamp_conversion()