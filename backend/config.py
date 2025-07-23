from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # Polygon.io API
    polygon_api_key: str
    
    # Alpaca API
    alpaca_paper_api_key: str
    alpaca_paper_secret_key: str
    alpaca_paper_base_url: str
    alpaca_live_api_key: str
    alpaca_live_secret_key: str
    alpaca_live_base_url: str
    
    # Database
    database_host: str
    database_port: int
    database_name: str
    database_user: str
    database_password: str
    
    # Trading
    trading_mode: Literal["paper", "live"] = "paper"
    
    # System
    log_level: str = "INFO"
    max_positions: int = 20
    max_position_size: float = 0.05
    daily_loss_limit: float = 0.02
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()