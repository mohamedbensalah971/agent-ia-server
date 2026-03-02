"""
Configuration Management for AI Agent Server
Handles environment variables and application settings
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """
    Application Settings
    
    All sensitive data should be in .env file
    """
    
    # API Configuration
    APP_NAME: str = "AI Test Automation Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Groq API
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS: int = 4000
    GROQ_TEMPERATURE: float = 0.1  # Low temperature for consistent code generation
    
    # Rate Limiting (Groq Free Tier)
    RATE_LIMIT_TOKENS_PER_MINUTE: int = 6000
    RATE_LIMIT_TOKENS_PER_DAY: int = 14400
    
    # Git Configuration
    GIT_REPO_PATH: str = "C:/Users/Stayha/Desktop/pfe/SmartTalk-Android"
    GIT_BRANCH_PREFIX: str = "fix/ai-correction"
    GIT_AUTHOR_NAME: str = "AI Agent"
    GIT_AUTHOR_EMAIL: str = "ai-agent@smarttalk.com"
    
    # Jenkins Configuration
    JENKINS_URL: Optional[str] = None
    JENKINS_USERNAME: Optional[str] = None
    JENKINS_TOKEN: Optional[str] = None
    
    # Cache Configuration
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/agent_ia.log"
    
    # RAG Configuration (for later integration)
    RAG_ENABLED: bool = False
    RAG_ENDPOINT: Optional[str] = "http://localhost:8001"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings
    
    Returns:
        Settings instance
    """
    return settings


# Validation on startup
def validate_settings():
    """
    Validate critical settings on startup
    """
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY must be set in environment variables or .env file")
    
    if not os.path.exists(settings.GIT_REPO_PATH):
        raise ValueError(f"Git repository not found at: {settings.GIT_REPO_PATH}")
    
    print("✅ Configuration validated successfully")


if __name__ == "__main__":
    # Test configuration
    validate_settings()
    print(f"App Name: {settings.APP_NAME}")
    print(f"Groq Model: {settings.GROQ_MODEL}")
    print(f"Git Repo: {settings.GIT_REPO_PATH}")
