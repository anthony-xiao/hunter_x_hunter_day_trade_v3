#!/usr/bin/env python3
"""
Startup script for the Algorithmic Trading System
This script handles initialization, dependency checks, and system startup
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta, timezone

# Add backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")

def install_dependencies():
    """Install required Python packages"""
    logger.info("Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found")
        sys.exit(1)
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        sys.exit(1)

def check_environment_variables():
    """Check if required environment variables are set"""
    logger.info("Checking environment variables...")
    
    env_file = Path(__file__).parent / ".env"
    
    if not env_file.exists():
        logger.error(".env file not found. Please create it with required API keys and database settings.")
        logger.info("Required variables:")
        logger.info("- POLYGON_API_KEY")
        logger.info("- ALPACA_API_KEY_PAPER")
        logger.info("- ALPACA_SECRET_KEY_PAPER")
        logger.info("- ALPACA_API_KEY_LIVE")
        logger.info("- ALPACA_SECRET_KEY_LIVE")
        logger.info("- DATABASE_HOST")
        logger.info("- DATABASE_PORT")
        logger.info("- DATABASE_NAME")
        logger.info("- DATABASE_USER")
        logger.info("- DATABASE_PASSWORD")
        sys.exit(1)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    required_vars = [
        'POLYGON_API_KEY',
        'ALPACA_API_KEY_PAPER',
        'ALPACA_SECRET_KEY_PAPER',
        'DATABASE_HOST',
        'DATABASE_PORT',
        'DATABASE_NAME',
        'DATABASE_USER',
        'DATABASE_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    logger.info("Environment variables check passed")

def check_database_connection():
    """Check database connection"""
    logger.info("Checking database connection...")
    
    try:
        from database import test_connection
        if test_connection():
            logger.info("Database connection successful")
        else:
            logger.error("Database connection failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        logger.info("Please check your database configuration in .env file")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    logger.info("Creating directories...")
    
    directories = [
        "logs",
        "models",
        "data/cache",
        "backups"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Directories created")

async def initialize_system():
    """Initialize the trading system"""
    logger.info("Initializing trading system...")
    
    try:
        # Import system components
        from database import create_tables
        from data.data_pipeline import DataPipeline
        from ml.model_trainer import ModelTrainer
        
        # Create database tables
        logger.info("Creating database tables...")
        create_tables()
        
        # Initialize data pipeline
        logger.info("Initializing data pipeline...")
        data_pipeline = DataPipeline()
        await data_pipeline.initialize_database()
        
        # Check if we have any historical data
        logger.info("Checking for historical data...")
        universe = await data_pipeline.update_trading_universe()
        
        # Download initial data if needed
        for symbol in universe[:3]:  # Start with first 3 symbols
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=30)
            
            df = await data_pipeline.load_market_data(symbol, start_date, end_date)
            
            if df.empty:
                logger.info(f"Downloading initial data for {symbol}...")
                await data_pipeline.download_historical_data(symbol, start_date, end_date)
        
        # Initialize ML models
        logger.info("Initializing ML models...")
        model_trainer = ModelTrainer(feature_count=50, create_model_dir=False)
        await model_trainer.load_models()
        
        logger.info("System initialization complete")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

def start_server():
    """Start the FastAPI server"""
    logger.info("Starting FastAPI server...")
    
    try:
        import uvicorn
        from main import app
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    logger.info("=" * 60)
    logger.info("ALGORITHMIC TRADING SYSTEM STARTUP")
    logger.info("=" * 60)
    
    try:
        # Pre-flight checks
        check_python_version()
        
        # Install dependencies
        install_dependencies()
        
        # Check environment
        check_environment_variables()
        
        # Create directories
        create_directories()
        
        # Check database
        check_database_connection()
        
        # Initialize system
        logger.info("Running system initialization...")
        asyncio.run(initialize_system())
        
        # Start server
        logger.info("All checks passed. Starting server...")
        logger.info("API will be available at: http://localhost:8000")
        logger.info("API documentation at: http://localhost:8000/docs")
        logger.info("=" * 60)
        
        start_server()
        
    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    main()