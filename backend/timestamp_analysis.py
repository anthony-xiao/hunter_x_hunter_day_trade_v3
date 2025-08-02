#!/usr/bin/env python3
"""
Timestamp Analysis Script

This script analyzes timestamps in our database and compares them with
what Polygon API actually sends us for a full trading day.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_pipeline import DataPipeline
from data.polygon_websocket import PolygonWebSocketManager
from database.models import MarketData, Features
from database import db_manager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd


class TimestampAnalyzer:
    def __init__(self):
        self.engine = db_manager.get_engine()
        self.Session = db_manager.SessionLocal
        self.data_pipeline = DataPipeline()
        
    def analyze_database_timestamps(self, symbol: str = "AAPL", days_back: int = 5):
        """
        Analyze timestamps in our database for market_data and features tables
        """
        print(f"\n=== DATABASE TIMESTAMP ANALYSIS for {symbol} (last {days_back} days) ===")
        
        with self.Session() as session:
            # Analyze market_data timestamps
            print("\n--- MARKET DATA TIMESTAMPS ---")
            market_query = text("""
                SELECT 
                    DATE(timestamp) as date,
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp,
                    COUNT(*) as record_count,
                    EXTRACT(HOUR FROM MIN(timestamp)) as first_hour,
                    EXTRACT(HOUR FROM MAX(timestamp)) as last_hour
                FROM market_data 
                WHERE symbol = :symbol 
                    AND timestamp >= NOW() - INTERVAL ':days days'
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """)
            
            result = session.execute(market_query, {"symbol": symbol, "days": days_back})
            market_data = result.fetchall()
            
            for row in market_data:
                print(f"Date: {row.date}")
                print(f"  First: {row.first_timestamp} (Hour: {row.first_hour})")
                print(f"  Last:  {row.last_timestamp} (Hour: {row.last_hour})")
                print(f"  Records: {row.record_count}")
                print()
            
            # Analyze features timestamps
            print("\n--- FEATURES TIMESTAMPS ---")
            features_query = text("""
                SELECT 
                    DATE(timestamp) as date,
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp,
                    COUNT(*) as record_count,
                    EXTRACT(HOUR FROM MIN(timestamp)) as first_hour,
                    EXTRACT(HOUR FROM MAX(timestamp)) as last_hour
                FROM features 
                WHERE symbol = :symbol 
                    AND timestamp >= NOW() - INTERVAL ':days days'
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """)
            
            result = session.execute(features_query, {"symbol": symbol, "days": days_back})
            features_data = result.fetchall()
            
            for row in features_data:
                print(f"Date: {row.date}")
                print(f"  First: {row.first_timestamp} (Hour: {row.first_hour})")
                print(f"  Last:  {row.last_timestamp} (Hour: {row.last_hour})")
                print(f"  Records: {row.record_count}")
                print()
    
    def analyze_timezone_info(self, symbol: str = "AAPL"):
        """
        Check timezone information in our database
        """
        print(f"\n=== TIMEZONE ANALYSIS ===")
        
        with self.Session() as session:
            # Check database timezone
            db_tz_query = text("SELECT current_setting('timezone') as db_timezone")
            result = session.execute(db_tz_query)
            db_tz = result.fetchone()
            print(f"Database timezone: {db_tz.db_timezone}")
            
            # Check sample timestamps with timezone info
            sample_query = text("""
                SELECT 
                    timestamp,
                    timestamp AT TIME ZONE 'UTC' as utc_time,
                    timestamp AT TIME ZONE 'America/New_York' as ny_time
                FROM market_data 
                WHERE symbol = :symbol 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            
            result = session.execute(sample_query, {"symbol": symbol})
            samples = result.fetchall()
            
            print("\nSample timestamps (latest 5):")
            for row in samples:
                print(f"  Raw: {row.timestamp}")
                print(f"  UTC: {row.utc_time}")
                print(f"  NY:  {row.ny_time}")
                print()
    
    async def fetch_polygon_sample(self, symbol: str = "AAPL"):
        """
        Fetch a sample day from Polygon to see what timestamps they send
        """
        print(f"\n=== POLYGON API SAMPLE for {symbol} ===")
        
        # Get yesterday's date for a complete trading day
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")
        
        try:
            # Use the data pipeline to fetch data
            print(f"Fetching Polygon data for {date_str}...")
            
            # This would typically be done through the REST API
            # Let's check what we have in our database for comparison
            with self.Session() as session:
                query = text("""
                    SELECT 
                        timestamp,
                        open, high, low, close, volume
                    FROM market_data 
                    WHERE symbol = :symbol 
                        AND DATE(timestamp) = :date
                    ORDER BY timestamp
                    LIMIT 10
                """)
                
                result = session.execute(query, {"symbol": symbol, "date": date_str})
                data = result.fetchall()
                
                if data:
                    print(f"\nSample data from our database for {date_str}:")
                    for i, row in enumerate(data):
                        print(f"  {i+1}. {row.timestamp} - O:{row.open} H:{row.high} L:{row.low} C:{row.close} V:{row.volume}")
                else:
                    print(f"No data found in database for {date_str}")
                    
        except Exception as e:
            print(f"Error fetching Polygon data: {e}")
    
    def check_trading_hours(self, symbol: str = "AAPL"):
        """
        Check if our timestamps align with expected trading hours
        """
        print(f"\n=== TRADING HOURS ANALYSIS for {symbol} ===")
        
        with self.Session() as session:
            # Check hour distribution
            hour_query = text("""
                SELECT 
                    EXTRACT(HOUR FROM timestamp) as hour,
                    COUNT(*) as count,
                    MIN(DATE(timestamp)) as first_date,
                    MAX(DATE(timestamp)) as last_date
                FROM market_data 
                WHERE symbol = :symbol 
                    AND timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY EXTRACT(HOUR FROM timestamp)
                ORDER BY hour
            """)
            
            result = session.execute(hour_query, {"symbol": symbol})
            hour_data = result.fetchall()
            
            print("Hour distribution (last 7 days):")
            for row in hour_data:
                print(f"  Hour {int(row.hour):02d}: {row.count} records ({row.first_date} to {row.last_date})")
            
            # Expected trading hours:
            # Pre-market: 4:00 AM - 9:30 AM ET
            # Regular: 9:30 AM - 4:00 PM ET  
            # After-hours: 4:00 PM - 8:00 PM ET
            
            print("\nExpected trading hours (ET):")
            print("  Pre-market: 04:00 - 09:30")
            print("  Regular:    09:30 - 16:00")
            print("  After-hours: 16:00 - 20:00")
            
            # If we're storing in UTC, add 5 hours (EST) or 4 hours (EDT)
            print("\nExpected hours in UTC (assuming EST +5):")
            print("  Pre-market: 09:00 - 14:30")
            print("  Regular:    14:30 - 21:00")
            print("  After-hours: 21:00 - 01:00+1")
    
    async def run_full_analysis(self, symbol: str = "AAPL"):
        """
        Run complete timestamp analysis
        """
        print(f"Starting comprehensive timestamp analysis for {symbol}...")
        
        # Database analysis
        self.analyze_database_timestamps(symbol)
        self.analyze_timezone_info(symbol)
        self.check_trading_hours(symbol)
        
        # Polygon comparison
        await self.fetch_polygon_sample(symbol)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("\nKey things to check:")
        print("1. Are timestamps consistently starting at 16:00 (4 PM)?")
        print("2. Is this 16:00 in UTC or local time?")
        print("3. Does this align with market close (4 PM ET) or open (9:30 AM ET)?")
        print("4. Are we missing pre-market or regular trading hours?")


async def main():
    analyzer = TimestampAnalyzer()
    
    # You can change the symbol here
    symbol = "AAPL"
    
    await analyzer.run_full_analysis(symbol)


if __name__ == "__main__":
    asyncio.run(main())