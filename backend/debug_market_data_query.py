#!/usr/bin/env python3
"""
Debug script to investigate why only 26,000 AAPL records are being loaded
when there are 399.3K records in the Supabase market_data database.
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from config import Settings

# Load configuration
settings = Settings()

def get_supabase_client() -> Client:
    """Initialize Supabase client"""
    return create_client(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_service_role_key
    )

async def test_total_aapl_count(supabase: Client):
    """Test total count of AAPL records in database"""
    print("\n=== Testing Total AAPL Count ===")
    try:
        # Get total count of AAPL records
        response = supabase.table('market_data').select(
            'id', count='exact'
        ).eq('symbol', 'AAPL').execute()
        
        total_count = response.count if hasattr(response, 'count') else len(response.data)
        print(f"Total AAPL records in database: {total_count}")
        
        # Also try without count to see actual data
        response_data = supabase.table('market_data').select(
            'timestamp'
        ).eq('symbol', 'AAPL').limit(10).execute()
        
        print(f"Sample timestamps (first 10):")
        for record in response_data.data[:5]:
            print(f"  {record['timestamp']}")
            
        return total_count
        
    except Exception as e:
        print(f"Error getting total count: {e}")
        return 0

async def test_date_range_query(supabase: Client, symbol: str, start_date: datetime, end_date: datetime):
    """Test the exact same query used in _load_market_data_chunk"""
    print(f"\n=== Testing Date Range Query ===")
    print(f"Symbol: {symbol}")
    print(f"Start: {start_date.isoformat()}")
    print(f"End: {end_date.isoformat()}")
    
    try:
        # Exact same query as in _load_market_data_chunk
        response = supabase.table('market_data').select(
            'timestamp, open, high, low, close, volume, vwap, transactions'
        ).eq('symbol', symbol).gte(
            'timestamp', start_date.isoformat()
        ).lte(
            'timestamp', end_date.isoformat()
        ).order('timestamp').execute()
        
        print(f"Records returned: {len(response.data)}")
        
        if response.data:
            print(f"First record: {response.data[0]['timestamp']}")
            print(f"Last record: {response.data[-1]['timestamp']}")
        else:
            print("No data returned")
            
        return len(response.data)
        
    except Exception as e:
        print(f"Error in date range query: {e}")
        return 0

async def test_pagination_limits(supabase: Client):
    """Test if there are pagination limits affecting results"""
    print("\n=== Testing Pagination Limits ===")
    
    try:
        # Test different limit values
        for limit in [1000, 5000, 10000, 50000, 100000]:
            response = supabase.table('market_data').select(
                'timestamp'
            ).eq('symbol', 'AAPL').limit(limit).execute()
            
            print(f"Limit {limit}: Got {len(response.data)} records")
            
            # If we get less than the limit, we've hit the actual data limit
            if len(response.data) < limit:
                print(f"  -> Actual data limit reached at {len(response.data)} records")
                break
                
    except Exception as e:
        print(f"Error testing pagination: {e}")

async def test_chunk_queries(supabase: Client):
    """Test the chunked query approach used in load_market_data"""
    print("\n=== Testing Chunked Queries (Same as load_market_data) ===")
    
    # Same date range as in the logs
    start_date = datetime(2023, 7, 15, 2, 44, 36, tzinfo=timezone.utc)
    end_date = datetime(2025, 8, 13, 2, 44, 36, tzinfo=timezone.utc)
    chunk_days = 30
    
    print(f"Total range: {start_date.date()} to {end_date.date()}")
    print(f"Chunk size: {chunk_days} days")
    
    total_records = 0
    chunk_count = 0
    current_start = start_date
    
    while current_start < end_date and chunk_count < 5:  # Limit to first 5 chunks for testing
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        
        print(f"\nChunk {chunk_count + 1}: {current_start.date()} to {current_end.date()}")
        
        chunk_records = await test_date_range_query(supabase, 'AAPL', current_start, current_end)
        total_records += chunk_records
        
        current_start = current_end
        chunk_count += 1
    
    print(f"\nTotal records from {chunk_count} chunks: {total_records}")
    return total_records

async def test_timestamp_formats(supabase: Client):
    """Test different timestamp formats to see if there's a format issue"""
    print("\n=== Testing Timestamp Formats ===")
    
    # Get a sample of timestamps from the database
    response = supabase.table('market_data').select(
        'timestamp'
    ).eq('symbol', 'AAPL').limit(5).execute()
    
    print("Sample timestamps from database:")
    for record in response.data:
        print(f"  {record['timestamp']}")
    
    # Test different query formats
    test_date = datetime(2025, 6, 16, tzinfo=timezone.utc)
    
    formats_to_test = [
        test_date.isoformat(),
        test_date.strftime('%Y-%m-%d %H:%M:%S+00'),
        test_date.strftime('%Y-%m-%d'),
    ]
    
    for fmt in formats_to_test:
        try:
            response = supabase.table('market_data').select(
                'timestamp'
            ).eq('symbol', 'AAPL').gte('timestamp', fmt).limit(5).execute()
            
            print(f"Format '{fmt}': {len(response.data)} records")
            
        except Exception as e:
            print(f"Format '{fmt}': Error - {e}")

async def test_actual_date_coverage(supabase: Client):
    """Test what date ranges actually have data"""
    print("\n=== Testing Actual Date Coverage ===")
    
    try:
        # Get min and max timestamps for AAPL
        response = supabase.table('market_data').select(
            'timestamp'
        ).eq('symbol', 'AAPL').order('timestamp', desc=False).limit(1).execute()
        
        min_timestamp = response.data[0]['timestamp'] if response.data else None
        
        response = supabase.table('market_data').select(
            'timestamp'
        ).eq('symbol', 'AAPL').order('timestamp', desc=True).limit(1).execute()
        
        max_timestamp = response.data[0]['timestamp'] if response.data else None
        
        print(f"Actual data range for AAPL:")
        print(f"  Min timestamp: {min_timestamp}")
        print(f"  Max timestamp: {max_timestamp}")
        
        if min_timestamp and max_timestamp:
            # Calculate the actual coverage
            min_dt = datetime.fromisoformat(min_timestamp.replace('Z', '+00:00'))
            max_dt = datetime.fromisoformat(max_timestamp.replace('Z', '+00:00'))
            actual_days = (max_dt - min_dt).days
            print(f"  Actual coverage: {actual_days} days")
            
            # Compare with requested range
            requested_start = datetime(2023, 7, 15, 2, 44, 36, tzinfo=timezone.utc)
            requested_end = datetime(2025, 8, 13, 2, 44, 36, tzinfo=timezone.utc)
            requested_days = (requested_end - requested_start).days
            
            print(f"  Requested range: {requested_days} days")
            print(f"  Coverage overlap: {min_dt >= requested_start and max_dt <= requested_end}")
            
    except Exception as e:
        print(f"Error testing date coverage: {e}")

async def main():
    """Main debug function"""
    print("=== Market Data Query Debug Script ===")
    print(f"Investigating why only 26,000 AAPL records are loaded when 399.3K exist")
    
    # Initialize Supabase client
    supabase = get_supabase_client()
    
    # Run all tests
    await test_total_aapl_count(supabase)
    await test_actual_date_coverage(supabase)
    await test_timestamp_formats(supabase)
    await test_pagination_limits(supabase)
    await test_chunk_queries(supabase)
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    asyncio.run(main())