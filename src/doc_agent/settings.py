"""FIXED — typed settings from environment (secrets live here, never in code/config)."""
from __future__ import annotations
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Load secrets from .env file (never commit .env to git).
    
    Usage:
        from doc_agent.settings import settings
        api_key = settings.llm_api_key
    """
    # LLM credentials
    llm_api_key: str = ""
    
    # Experiment tracking
    wandb_api_key: str = ""
    
    # Kaggle dataset download
    kaggle_username: str = ""
    kaggle_api_key: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()  # import this; do not read os.environ elsewhere
