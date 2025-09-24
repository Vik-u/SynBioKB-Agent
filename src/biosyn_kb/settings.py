from typing import Literal, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables.

    Environment variables:
    - SEARCH_PROVIDER: one of ["serpapi", "brave"] (default: "serpapi")
    - SERPAPI_API_KEY: API key for SerpAPI
    - BRAVE_API_KEY: API key for Brave Search (optional)
    - API_CONFIG_PATH: optional path to YAML file with API keys
    - USER_AGENT: HTTP user agent (default: "biosyn-kb-agent/0.1")
    - REQUEST_TIMEOUT: request timeout in seconds (default: 15.0)
    """

    search_provider: Literal["serpapi", "brave"] = "serpapi"
    serpapi_api_key: Optional[str] = None
    brave_api_key: Optional[str] = None
    api_config_path: Optional[str] = None

    user_agent: str = "biosyn-kb-agent/0.1"
    request_timeout: float = 15.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
