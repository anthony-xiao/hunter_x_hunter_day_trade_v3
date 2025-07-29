#!/usr/bin/env python3
"""
Script to check available market data in the database
"""

import sys
import os
from datetime import datetime, timedelta, timezone
from sqlalchemy import text

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import db_manager

def check_market_data():
    """Check what market data is available in the database"""
    try:
        with db_manager.get_session() as session:
            # Get overall statistics
            result = session.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(timestamp) as earliest_timestamp,
                    MAX(timestamp) as latest_timestamp
                FROM market_data
            """))
            
            stats = result.fetchone()
            
            print("Market Data Overview:")
            print(f"Total Records: {stats.total_records}")
            print(f"Unique Symbols: {stats.unique_symbols}")
            print(f"Earliest Data: {stats.earliest_timestamp}")
            print(f"Latest Data: {stats.latest_timestamp}")
            
            if stats.total_records == 0:
                print("\nNo market data found in database.")
                return
            
            # Get symbols with recent data (last 24 hours)
            recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            result = session.execute(text("""
                SELECT 
                    symbol,
                    COUNT(*) as record_count,
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM market_data
                WHERE timestamp >= :recent_cutoff
                GROUP BY symbol
                ORDER BY record_count DESC
            """), {'recent_cutoff': recent_cutoff})
            
            recent_data = result.fetchall()
            
            print(f"\nSymbols with data in last 24 hours ({len(recent_data)} symbols):")
            for row in recent_data:
                print(f"  {row.symbol}: {row.record_count} records ({row.earliest} to {row.latest})")
            
            # Get symbols with data in last hour
            hour_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            result = session.execute(text("""
                SELECT 
                    symbol,
                    COUNT(*) as record_count,
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM market_data
                WHERE timestamp >= :hour_cutoff
                GROUP BY symbol
                ORDER BY record_count DESC
            """), {'hour_cutoff': hour_cutoff})
            
            hour_data = result.fetchall()
            
            print(f"\nSymbols with data in last hour ({len(hour_data)} symbols):")
            if hour_data:
                for row in hour_data:
                    print(f"  {row.symbol}: {row.record_count} records ({row.earliest} to {row.latest})")
            else:
                print("  No data in the last hour.")
            
            # Sample some recent data
            result = session.execute(text("""
                SELECT symbol, timestamp, open, high, low, close, volume, vwap, transactions
                FROM market_data
                ORDER BY timestamp DESC
                LIMIT 10
            """))
            
            sample_data = result.fetchall()
            
            print("\nSample of most recent data:")
            for row in sample_data:
                print(f"  {row.symbol} @ {row.timestamp}: O={row.open} H={row.high} L={row.low} C={row.close} V={row.volume}")
            
    except Exception as e:
        print(f"Error checking market data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_market_data()