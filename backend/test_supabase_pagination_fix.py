#!/usr/bin/env python3
"""
Test proper Supabase pagination using multiple queries with timestamp-based pagination.
"""

import asyncio
from datetime import datetime, timezone
from supabase import create_client
from config import Settings

settings = Settings()
supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

async def test_timestamp_based_pagination():
    """Test timestamp-based pagination to get all records"""
    print("=== Testing Timestamp-Based Pagination ===")
    
    # Test date range with known data
    start_date = datetime(2025, 6, 16, tzinfo=timezone.utc)
    end_date = datetime(2025, 6, 20, tzinfo=timezone.utc)
    
    print(f"Testing date range: {start_date.date()} to {end_date.date()}")
    
    # First get the count to know what we expect
    response = supabase.table('market_data').select(
        '*', count='exact'
    ).eq('symbol', 'AAPL').gte(
        'timestamp', start_date.isoformat()
    ).lte(
        'timestamp', end_date.isoformat()
    ).execute()
    
    expected_count = response.count if hasattr(response, 'count') else 0
    print(f"Expected total records: {expected_count}")
    
    # Method: Use timestamp-based pagination
    print("\nUsing timestamp-based pagination:")
    
    all_data = []
    current_start = start_date
    page_count = 0
    
    while current_start < end_date:
        page_count += 1
        
        # Get 1000 records starting from current_start
        response = supabase.table('market_data').select(
            'timestamp, open, high, low, close, volume, vwap, transactions'
        ).eq('symbol', 'AAPL').gte(
            'timestamp', current_start.isoformat()
        ).lte(
            'timestamp', end_date.isoformat()
        ).order('timestamp').limit(1000).execute()
        
        if not response.data:
            print(f"  Page {page_count}: No more data")
            break
        
        print(f"  Page {page_count}: {len(response.data)} records")
        print(f"    First: {response.data[0]['timestamp']}")
        print(f"    Last:  {response.data[-1]['timestamp']}")
        
        all_data.extend(response.data)
        
        # If we got less than 1000 records, we're done
        if len(response.data) < 1000:
            print(f"  Page {page_count}: Reached end (got {len(response.data)} < 1000)")
            break
        
        # Update current_start to be just after the last timestamp
        last_timestamp = response.data[-1]['timestamp']
        # Parse the timestamp and add 1 second to avoid duplicates
        last_dt = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
        current_start = last_dt + timedelta(seconds=1)
        
        print(f"    Next start: {current_start.isoformat()}")
        
        # Safety limit
        if page_count > 20:
            print("  Safety limit reached (20 pages)")
            break
        
        # Small delay
        await asyncio.sleep(0.1)
    
    print(f"\nTotal records retrieved: {len(all_data)}")
    print(f"Expected: {expected_count}")
    print(f"Success rate: {len(all_data) / expected_count * 100:.1f}%" if expected_count > 0 else "N/A")
    
    return len(all_data)

if __name__ == "__main__":
    from datetime import timedelta
    asyncio.run(test_timestamp_based_pagination())