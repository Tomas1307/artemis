from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables.

    All values are read from the .env file in the project root.
    Access via the module-level `settings` singleton.

    Attributes:
        NVIDIA_BASE_URL: Base URL for the NVIDIA API endpoint.
        NVIDIA_MODEL: Model identifier for generation calls (Devstral-2-123B).
        NVIDIA_API_KEY: API key for generation model authentication.
        NVIDIA_JUDGE_MODEL: Model identifier for validation/judge calls (QwQ-32B).
        NVIDIA_JUDGE_API_KEY: API key for judge model authentication.
        SKELETON_PATH: Path to the MASA universe skeleton YAML file.
    """

    NVIDIA_BASE_URL: str
    NVIDIA_MODEL: str
    NVIDIA_API_KEY: str
    NVIDIA_JUDGE_MODEL: str
    NVIDIA_JUDGE_API_KEY: str
    SKELETON_PATH: str = str(Path(__file__).parent / "skeleton" / "skeleton.yaml")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
