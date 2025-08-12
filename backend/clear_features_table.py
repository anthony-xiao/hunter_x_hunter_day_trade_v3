#!/usr/bin/env python3
"""
Script to clear the features table in Supabase database.
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
        # Get Supabase client
        from database import db_manager
        supabase = db_manager.get_supabase_client()
        
        if not supabase:
            logger.error("Supabase client not available")
            return
        # Get count before deletion
        count_result = supabase.table('features').select('id', count='exact').execute()
        initial_count = count_result.count
        
        logger.info(f"Found {initial_count} records in features table")
        
        if initial_count == 0:
            logger.info("Features table is already empty")
            return
        
        # Clear the features table
        logger.info("Clearing features table...")
        delete_result = supabase.table('features').delete().neq('id', 0).execute()
        
        # Verify deletion
        count_result = supabase.table('features').select('id', count='exact').execute()
        final_count = count_result.count
        
        logger.info(f"Successfully cleared features table. Deleted {initial_count} records.")
        logger.info(f"Features table now has {final_count} records")
        
        # Note: Sequence reset is handled automatically by Supabase
        logger.info("Features table sequence will be managed by Supabase automatically.")
            
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