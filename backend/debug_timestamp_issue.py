#!/usr/bin/env python3
"""
Debug script to verify timestamp timezone consistency between market_data and features tables.
This script will help identify and fix the timezone mismatch issue.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database.models import Base
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from loguru import logger
import pandas as pd

# Database configuration
DATABASE_URL = "postgresql://postgres:password@localhost:5432/algo_trading"

def create_db_engine():
    """Create database engine"""
    return create_engine(DATABASE_URL, echo=False)

async def check_timestamp_consistency():
    """Check timestamp consistency between market_data and features tables"""
    engine = create_db_engine()
    Session = sessionmaker(bind=engine)
    
    try:
        with Session() as session:
            # Check market_data timestamps
            logger.info("Checking market_data timestamps...")
            market_result = session.execute(text("""
                SELECT symbol, timestamp, 
                       EXTRACT(timezone FROM timestamp) as tz_offset,
                       timestamp AT TIME ZONE 'UTC' as utc_timestamp
                FROM market_data 
                WHERE symbol IN ('AAPL', 'NVDA')
                ORDER BY timestamp DESC 
                LIMIT 10
            """))
            
            market_data = market_result.fetchall()
            logger.info(f"Found {len(market_data)} market_data records")
            
            for row in market_data:
                logger.info(f"Market Data - Symbol: {row[0]}, Timestamp: {row[1]}, TZ Offset: {row[2]}, UTC: {row[3]}")
            
            # Check features timestamps
            logger.info("\nChecking features timestamps...")
            features_result = session.execute(text("""
                SELECT symbol, timestamp,
                       EXTRACT(timezone FROM timestamp) as tz_offset,
                       timestamp AT TIME ZONE 'UTC' as utc_timestamp
                FROM features 
                WHERE symbol IN ('AAPL', 'NVDA')
                ORDER BY timestamp DESC 
                LIMIT 10
            """))
            
            features_data = features_result.fetchall()
            logger.info(f"Found {len(features_data)} features records")
            
            for row in features_data:
                logger.info(f"Features - Symbol: {row[0]}, Timestamp: {row[1]}, TZ Offset: {row[2]}, UTC: {row[3]}")
            
            # Check for timestamp mismatches
            logger.info("\nChecking for timestamp mismatches...")
            mismatch_result = session.execute(text("""
                SELECT 
                    f.symbol,
                    f.timestamp as feature_timestamp,
                    m.timestamp as market_timestamp,
                    f.timestamp - m.timestamp as time_diff
                FROM features f
                LEFT JOIN market_data m ON f.symbol = m.symbol 
                    AND ABS(EXTRACT(EPOCH FROM (f.timestamp - m.timestamp))) < 300  -- Within 5 minutes
                WHERE f.symbol IN ('AAPL', 'NVDA')
                    AND m.timestamp IS NULL
                ORDER BY f.timestamp DESC
                LIMIT 20
            """))
            
            mismatches = mismatch_result.fetchall()
            logger.info(f"Found {len(mismatches)} potential timestamp mismatches")
            
            for row in mismatches:
                logger.warning(f"Mismatch - Symbol: {row[0]}, Feature TS: {row[1]}, Market TS: {row[2]}, Diff: {row[3]}")
            
            # Check for "future" timestamps
            logger.info("\nChecking for future timestamps...")
            now_utc = datetime.now(timezone.utc)
            future_result = session.execute(text("""
                SELECT 'features' as table_name, symbol, timestamp
                FROM features 
                WHERE timestamp > :now_utc
                UNION ALL
                SELECT 'market_data' as table_name, symbol, timestamp
                FROM market_data 
                WHERE timestamp > :now_utc
                ORDER BY timestamp DESC
            """), {'now_utc': now_utc})
            
            future_timestamps = future_result.fetchall()
            logger.info(f"Found {len(future_timestamps)} future timestamps")
            
            for row in future_timestamps:
                logger.warning(f"Future timestamp - Table: {row[0]}, Symbol: {row[1]}, Timestamp: {row[2]}")
            
            # Summary statistics
            logger.info("\nSummary Statistics:")
            
            # Count records by hour for features
            hourly_features = session.execute(text("""
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(*) as feature_count
                FROM features 
                WHERE symbol IN ('AAPL', 'NVDA')
                    AND timestamp >= :start_time
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour DESC
                LIMIT 24
            """), {'start_time': now_utc - timedelta(days=1)})
            
            logger.info("Features by hour (last 24 hours):")
            for row in hourly_features.fetchall():
                logger.info(f"  {row[0]}: {row[1]} features")
            
            # Count records by hour for market data
            hourly_market = session.execute(text("""
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(*) as market_count
                FROM market_data 
                WHERE symbol IN ('AAPL', 'NVDA')
                    AND timestamp >= :start_time
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour DESC
                LIMIT 24
            """), {'start_time': now_utc - timedelta(days=1)})
            
            logger.info("Market data by hour (last 24 hours):")
            for row in hourly_market.fetchall():
                logger.info(f"  {row[0]}: {row[1]} market records")
                
    except Exception as e:
        logger.error(f"Error checking timestamp consistency: {e}")
        raise

async def fix_timezone_issues():
    """Fix timezone issues in existing data (if needed)"""
    engine = create_db_engine()
    Session = sessionmaker(bind=engine)
    
    logger.info("Checking if timezone fixes are needed...")
    
    try:
        with Session() as session:
            # Check if we have timezone-naive timestamps that need fixing
            naive_features = session.execute(text("""
                SELECT COUNT(*) 
                FROM features 
                WHERE EXTRACT(timezone FROM timestamp) IS NULL
            """)).scalar()
            
            naive_market = session.execute(text("""
                SELECT COUNT(*) 
                FROM market_data 
                WHERE EXTRACT(timezone FROM timestamp) IS NULL
            """)).scalar()
            
            logger.info(f"Found {naive_features} timezone-naive features records")
            logger.info(f"Found {naive_market} timezone-naive market_data records")
            
            if naive_features > 0 or naive_market > 0:
                logger.warning("Found timezone-naive timestamps. Consider running timezone conversion if needed.")
                logger.info("Note: The updated code will handle timezone conversion for new data automatically.")
            else:
                logger.info("All timestamps appear to have timezone information.")
                
    except Exception as e:
        logger.error(f"Error checking timezone issues: {e}")
        raise

async def main():
    """Main function"""
    logger.info("Starting timestamp consistency check...")
    
    await check_timestamp_consistency()
    await fix_timezone_issues()
    
    logger.info("Timestamp consistency check completed.")
    logger.info("\nRecommendations:")
    logger.info("1. The updated code now ensures all timestamps are stored in UTC")
    logger.info("2. New market data and features will have consistent UTC timestamps")
    logger.info("3. Monitor the logs for any remaining timezone-related issues")
    logger.info("4. Consider re-running feature engineering for recent data if mismatches persist")

if __name__ == "__main__":
    asyncio.run(main())