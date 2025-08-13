#!/usr/bin/env python3
"""
Debug script to test different pagination approaches with Supabase.
"""

import asyncio
from datetime import datetime, timezone
from supabase import create_client
from config import Settings

settings = Settings()
supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

async def test_pagination_methods():
    """Test different pagination methods"""
    print("=== Testing Supabase Pagination Methods ===")
    
    # Test date range with known data
    start_date = datetime(2025, 6, 16, tzinfo=timezone.utc)
    end_date = datetime(2025, 6, 20, tzinfo=timezone.utc)  # 4 days
    
    print(f"Testing date range: {start_date.date()} to {end_date.date()}")
    
    # Method 1: Using range() with different approaches
    print("\n1. Testing range() method:")
    
    try:
        # Test basic range
        response = supabase.table('market_data').select(
            'timestamp'
        ).eq('symbol', 'AAPL').gte(
            'timestamp', start_date.isoformat()
        ).lte(
            'timestamp', end_date.isoformat()
        ).order('timestamp').range(0, 1999).execute()
        
        print(f"  Range 0-1999: {len(response.data)} records")
        
        # Test higher range
        response = supabase.table('market_data').select(
            'timestamp'
        ).eq('symbol', 'AAPL').gte(
            'timestamp', start_date.isoformat()
        ).lte(
            'timestamp', end_date.isoformat()
        ).order('timestamp').range(0, 4999).execute()
        
        print(f"  Range 0-4999: {len(response.data)} records")
        
        # Test even higher range
        response = supabase.table('market_data').select(
            'timestamp'
        ).eq('symbol', 'AAPL').gte(
            'timestamp', start_date.isoformat()
        ).lte(
            'timestamp', end_date.isoformat()
        ).order('timestamp').range(0, 9999).execute()
        
        print(f"  Range 0-9999: {len(response.data)} records")
        
    except Exception as e:
        print(f"  Error with range method: {e}")
    
    # Method 2: Using limit() with offset
    print("\n2. Testing limit() with multiple calls:")
    
    try:
        total_records = 0
        page_size = 1000
        page = 0
        
        while True:
            offset = page * page_size
            
            response = supabase.table('market_data').select(
                'timestamp'
            ).eq('symbol', 'AAPL').gte(
                'timestamp', start_date.isoformat()
            ).lte(
                'timestamp', end_date.isoformat()
            ).order('timestamp').limit(page_size).offset(offset).execute()
            
            if not response.data:
                break
                
            total_records += len(response.data)
            print(f"  Page {page + 1}: {len(response.data)} records (total: {total_records})")
            
            if len(response.data) < page_size:
                break
                
            page += 1
            
            if page > 10:  # Safety limit
                print("  Stopping at page 10 for safety")
                break
        
        print(f"  Total records with limit/offset: {total_records}")
        
    except Exception as e:
        print(f"  Error with limit/offset method: {e}")
    
    # Method 3: Test without any limits
    print("\n3. Testing without limits:")
    
    try:
        response = supabase.table('market_data').select(
            'timestamp'
        ).eq('symbol', 'AAPL').gte(
            'timestamp', start_date.isoformat()
        ).lte(
            'timestamp', end_date.isoformat()
        ).order('timestamp').execute()
        
        print(f"  No limits: {len(response.data)} records")
        
    except Exception as e:
        print(f"  Error without limits: {e}")
    
    # Method 4: Test count to see actual data available
    print("\n4. Testing count:")
    
    try:
        response = supabase.table('market_data').select(
            '*', count='exact'
        ).eq('symbol', 'AAPL').gte(
            'timestamp', start_date.isoformat()
        ).lte(
            'timestamp', end_date.isoformat()
        ).execute()
        
        count = response.count if hasattr(response, 'count') else 'unknown'
        print(f"  Actual count in date range: {count}")
        print(f"  Data returned: {len(response.data)}")
        
    except Exception as e:
        print(f"  Error with count: {e}")

if __name__ == "__main__":
    asyncio.run(test_pagination_methods())