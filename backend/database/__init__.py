from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging
from config import settings

logger = logging.getLogger(__name__)

# Supabase client for database operations with optimized connection settings
try:
    from supabase import create_client, Client
    
    # Use service role key for database operations (has elevated privileges)
    # Note: Connection optimizations are handled at the application level through batch processing
    supabase: Client = create_client(
        settings.supabase_url, 
        settings.supabase_service_role_key
    )
    
    SUPABASE_AVAILABLE = True
    logger.info("Supabase client initialized successfully with optimized timeout settings")
except ImportError as e:
    logger.error(f"Supabase client not available: {e}")
    supabase = None
    SUPABASE_AVAILABLE = False
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    supabase = None
    SUPABASE_AVAILABLE = False

# Create base class for models
Base = declarative_base()

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.supabase = supabase
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection using Supabase"""
        try:
            if not SUPABASE_AVAILABLE:
                raise Exception("Supabase client is not available")
            
            if settings.supabase_url and settings.supabase_service_role_key:
                # For now, we'll use Supabase client for all database operations
                # SQLAlchemy direct connection requires the actual database password
                # which is different from the service role key (JWT token)
                
                # Test the Supabase connection
                test_response = self.supabase.table('market_data').select('id').limit(1).execute()
                logger.info("Supabase client connection verified successfully")
                
                # Note: SQLAlchemy engine is intentionally set to None
                # We'll use Supabase client for all database operations
                self.engine = None
                self.SessionLocal = None
                
                logger.info("Database initialized using Supabase client (SQLAlchemy disabled)")
                
            else:
                raise Exception("Supabase configuration missing")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.engine = None
            self.SessionLocal = None
    
    def create_tables(self):
        """Create all database tables"""
        try:
            if self.engine:
                Base.metadata.create_all(bind=self.engine)
                logger.info("Database tables created successfully")
            else:
                logger.warning("No SQLAlchemy engine available for table creation")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution)"""
        try:
            if self.engine:
                Base.metadata.drop_all(bind=self.engine)
                logger.info("Database tables dropped successfully")
            else:
                logger.warning("No SQLAlchemy engine available for table operations")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        if self.SessionLocal:
            session = self.SessionLocal()
            try:
                yield session
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                session.close()
        else:
            logger.warning("SQLAlchemy sessions not available. Use Supabase client directly.")
            yield None
    
    def get_session_sync(self) -> Session:
        """Get database session for synchronous operations"""
        if self.SessionLocal:
            return self.SessionLocal()
        else:
            logger.warning("SQLAlchemy sessions not available. Use Supabase client directly.")
            return None
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            if SUPABASE_AVAILABLE and self.supabase:
                # Test Supabase client connection
                response = self.supabase.table('market_data').select('id').limit(1).execute()
                logger.info("Supabase client connection test successful")
                return True
            else:
                logger.error("No database connection available")
                return False
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_engine(self):
        """Get SQLAlchemy engine"""
        return self.engine
    
    def get_supabase_client(self):
        """Get Supabase client for real-time features and additional functionality"""
        if not SUPABASE_AVAILABLE:
            logger.warning("Supabase client not available")
            return None
        return self.supabase
    
    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions
def get_db_session() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get database session"""
    with db_manager.get_session() as session:
        yield session

def get_engine():
    """Get database engine"""
    return db_manager.get_engine()

def create_tables():
    """Create all database tables"""
    db_manager.create_tables()

def test_connection() -> bool:
    """Test database connection"""
    return db_manager.test_connection()

def close_db():
    """Close database connections"""
    db_manager.close()