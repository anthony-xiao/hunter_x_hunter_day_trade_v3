#!/usr/bin/env python3
"""
Script to clear the features table in PostgreSQL database.
This allows for regenerating features with the fixed timezone handling.
"""

import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from loguru import logger

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings

def clear_features_table():
    """Clear all records from the features table"""
    try:
        # Create database engine
        engine = create_engine(
            f"postgresql://{settings.database_user}:{settings.database_password}@{settings.database_host}:{settings.database_port}/{settings.database_name}"
        )
        
        # Create session
        Session = sessionmaker(bind=engine)
        
        with Session() as session:
            # Get count before deletion
            count_result = session.execute(text("SELECT COUNT(*) FROM features"))
            initial_count = count_result.scalar()
            
            logger.info(f"Found {initial_count} records in features table")
            
            if initial_count == 0:
                logger.info("Features table is already empty")
                return
            
            # Clear the features table
            logger.info("Clearing features table...")
            session.execute(text("DELETE FROM features"))
            session.commit()
            
            # Verify deletion
            count_result = session.execute(text("SELECT COUNT(*) FROM features"))
            final_count = count_result.scalar()
            
            logger.info(f"Successfully cleared features table. Deleted {initial_count} records.")
            logger.info(f"Features table now has {final_count} records")
            
            # Reset the sequence (optional, for clean ID numbering)
            logger.info("Resetting features table ID sequence...")
            session.execute(text("ALTER SEQUENCE features_id_seq RESTART WITH 1"))
            session.commit()
            
            logger.success("Features table cleared successfully! Ready for feature regeneration.")
            
    except Exception as e:
        logger.error(f"Failed to clear features table: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting features table cleanup...")
    clear_features_table()
    logger.info("Features table cleanup completed.")
    print("\n✅ Features table has been cleared successfully!")
    print("You can now regenerate features with the fixed timezone handling.")