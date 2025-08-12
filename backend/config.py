from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # Supabase settings - loaded from .env file
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_role_key: Optional[str] = None
    supabase_project_ref: Optional[str] = None
    
    # API Keys
    polygon_api_key: Optional[str] = None
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    
    # Alpaca Trading Configuration
    alpaca_paper_api_key: Optional[str] = None
    alpaca_paper_secret_key: Optional[str] = None
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets/v2"
    alpaca_live_api_key: Optional[str] = None
    alpaca_live_secret_key: Optional[str] = None
    alpaca_live_base_url: str = "https://api.alpaca.markets"
    trading_mode: str = "paper"
    
    class Config:
        # Use .env file in the backend directory
        env_file = Path(__file__).parent / ".env"

settings = Settings()