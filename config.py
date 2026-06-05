"""config.py — project-wide settings loaded from environment or .env file.

Uses pydantic-settings so any value can be overridden at runtime without
touching the code:

    OLLAMA_URL=http://remote-host:11434 python -m bench.benchmark

Or via a ``.env`` file in the project root.  ``extra="ignore"`` means
unrecognised environment variables are silently skipped rather than raising
a validation error.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the MRL benchmarker.

    All fields have sensible defaults so the project works out-of-the-box
    with a local Ollama install and no ``.env`` file.

    Attributes:
        ollama_url: Base URL of the Ollama server that serves the embedding
            model.  Must include scheme and port.  Default points to a local
            Ollama instance on the standard port.
        embed_model: Ollama model name to use for embeddings.  Must be
            an MRL-capable model; the benchmark is designed around
            ``nomic-embed-text`` which produces 768-d MRL vectors.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text"


settings = Settings()
