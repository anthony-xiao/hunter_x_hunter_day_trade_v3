#!/usr/bin/env python3
"""
Market Data Download Script

This script downloads historical market data for specified symbols and date ranges
using the DataPipeline class. It populates the market_data table with minute-level
data from Polygon.io.

Usage:
    python download_market_data.py --start-date 2024-01-01 --end-date 2024-12-31 --symbols AAPL,MSFT,GOOGL
    python download_market_data.py --start-date 2024-06-01 --end-date 2024-06-30 --symbols TSLA
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import List

# Import the DataPipeline class
from data.data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('download_market_data.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Download historical market data for specified symbols and date range',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_market_data.py --start-date 2024-01-01 --end-date 2024-12-31 --symbols AAPL,MSFT,GOOGL
  python download_market_data.py --start-date 2024-06-01 --end-date 2024-06-30 --symbols TSLA
  python download_market_data.py --start-date 2024-01-01 --end-date 2024-01-31 --symbols AAPL,MSFT,GOOGL,TSLA,AMZN
        """
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format (e.g., 2024-01-01)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date in YYYY-MM-DD format (e.g., 2024-12-31)'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        required=True,
        help='Comma-separated list of stock symbols (e.g., AAPL,MSFT,GOOGL)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force download even if data already exists with good coverage'
    )
    
    return parser.parse_args()


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime object"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


def parse_symbols(symbols_str: str) -> List[str]:
    """Parse comma-separated symbols string to list"""
    symbols = [symbol.strip().upper() for symbol in symbols_str.split(',')]
    # Remove empty strings
    symbols = [symbol for symbol in symbols if symbol]
    
    if not symbols:
        raise ValueError("No valid symbols provided")
    
    return symbols


async def download_data_for_symbol(pipeline: DataPipeline, symbol: str, start_date: datetime, end_date: datetime) -> bool:
    """Download data for a single symbol"""
    try:
        logger.info(f"Starting download for {symbol}...")
        
        # Download historical data
        df = await pipeline.download_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            logger.warning(f"No data downloaded for {symbol}")
            return False
        
        logger.info(f"Successfully downloaded {len(df)} data points for {symbol}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download data for {symbol}: {e}")
        return False


async def main():
    """Main function to orchestrate the download process"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Parse and validate inputs
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        symbols = parse_symbols(args.symbols)
        
        # Validate date range
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        logger.info(f"Starting market data download process...")
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Total symbols to process: {len(symbols)}")
        
        # Initialize DataPipeline
        logger.info("Initializing DataPipeline...")
        pipeline = DataPipeline()
        
        # Download data for each symbol
        successful_downloads = 0
        failed_downloads = 0
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Processing symbol {i}/{len(symbols)}: {symbol}")
            
            success = await download_data_for_symbol(
                pipeline=pipeline,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if success:
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            # Progress update
            logger.info(f"Progress: {i}/{len(symbols)} symbols processed")
        
        # Final summary
        logger.info("\n" + "="*50)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*50)
        logger.info(f"Total symbols processed: {len(symbols)}")
        logger.info(f"Successful downloads: {successful_downloads}")
        logger.info(f"Failed downloads: {failed_downloads}")
        logger.info(f"Success rate: {(successful_downloads/len(symbols)*100):.1f}%")
        
        if failed_downloads > 0:
            logger.warning(f"Some downloads failed. Check the logs above for details.")
        else:
            logger.info("All downloads completed successfully!")
        
        logger.info("Market data download process completed.")
        
    except KeyboardInterrupt:
        logger.info("Download process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())