#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_pipeline import DataPipeline

def check_nvda_data():
    """Check what NVDA data exists in the database"""
    try:
        dp = DataPipeline()
        
        # Check total count of NVDA records
        count_result = dp.supabase.table('market_data').select('*', count='exact').eq('symbol', 'NVDA').execute()
        print(f"Total NVDA records in database: {count_result.count}")
        
        if count_result.count > 0:
            # Get first few records
            first_records = dp.supabase.table('market_data').select('symbol, timestamp').eq('symbol', 'NVDA').order('timestamp').limit(5).execute()
            print(f"\nFirst 5 records:")
            for record in first_records.data:
                print(f"  {record['timestamp']}")
            
            # Get last few records
            last_records = dp.supabase.table('market_data').select('symbol, timestamp').eq('symbol', 'NVDA').order('timestamp', desc=True).limit(5).execute()
            print(f"\nLast 5 records:")
            for record in last_records.data:
                print(f"  {record['timestamp']}")
            
            # Check date range
            date_range = dp.supabase.table('market_data').select('timestamp').eq('symbol', 'NVDA').order('timestamp').limit(1).execute()
            earliest = date_range.data[0]['timestamp'] if date_range.data else None
            
            date_range = dp.supabase.table('market_data').select('timestamp').eq('symbol', 'NVDA').order('timestamp', desc=True).limit(1).execute()
            latest = date_range.data[0]['timestamp'] if date_range.data else None
            
            print(f"\nDate range: {earliest} to {latest}")
        else:
            print("No NVDA records found in database")
            
            # Check what symbols do exist
            symbols_result = dp.supabase.table('market_data').select('symbol', count='exact').execute()
            print(f"\nTotal records in market_data table: {symbols_result.count}")
            
            # Get unique symbols
            unique_symbols = dp.supabase.rpc('get_unique_symbols').execute()
            if unique_symbols.data:
                print(f"Available symbols: {unique_symbols.data}")
            else:
                # Fallback method
                sample_records = dp.supabase.table('market_data').select('symbol').limit(10).execute()
                if sample_records.data:
                    symbols = list(set([r['symbol'] for r in sample_records.data]))
                    print(f"Sample symbols found: {symbols}")
                else:
                    print("No data found in market_data table at all")
        
    except Exception as e:
        print(f"Error checking NVDA data: {e}")

if __name__ == "__main__":
    check_nvda_data()