"""Global data pipeline registry for consistent access across the application"""

import logging
from typing import Optional
from threading import Lock

logger = logging.getLogger(__name__)

class DataPipelineRegistry:
    """Singleton registry for global data pipeline access"""
    
    _instance: Optional['DataPipelineRegistry'] = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._data_pipeline = None
        return cls._instance
    
    def register_data_pipeline(self, data_pipeline):
        """Register the global data pipeline instance"""
        self._data_pipeline = data_pipeline
        logger.info("Data pipeline registered globally")
    
    def get_data_pipeline(self):
        """Get the global data pipeline instance"""
        if self._data_pipeline is None:
            # Try to create a new instance as fallback
            try:
                from data.data_pipeline import DataPipeline
                self._data_pipeline = DataPipeline()
                logger.info("Created fallback data pipeline instance")
            except Exception as e:
                logger.warning(f"Could not create fallback data pipeline: {e}")
                return None
        
        return self._data_pipeline
    
    def is_available(self) -> bool:
        """Check if data pipeline is available"""
        return self._data_pipeline is not None

# Global registry instance
_registry = DataPipelineRegistry()

def register_data_pipeline(data_pipeline):
    """Register the global data pipeline instance"""
    _registry.register_data_pipeline(data_pipeline)

def get_data_pipeline():
    """Get the global data pipeline instance"""
    return _registry.get_data_pipeline()

def is_data_pipeline_available() -> bool:
    """Check if data pipeline is available"""
    return _registry.is_available()