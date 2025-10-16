from functools import lru_cache
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class LlamaServerSettings(BaseSettings):
    """Runtime configuration for the LLaMA server."""

    model_id: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        description="Hugging Face model identifier to load at startup.",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Hostname the FastAPI server should bind to for remote access.",
    )
    port: int = Field(
        default=31511,
        description="Port the FastAPI server should listen on.",
    )
    max_input_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens accepted from a request history + prompt.",
    )
    max_new_tokens: int = Field(
        default=512,
        description="Default maximum number of new tokens to generate when not provided.",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature used as a default for generation.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p nucleus sampling threshold used as a default.",
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="Hugging Face access token (required for gated models).",
        env="HF_TOKEN",
    )
    device_map: Optional[str] = Field(
        default="auto",
        description="Device map strategy passed to Transformers when loading the model.",
    )
    torch_dtype: Optional[str] = Field(
        default="auto",
        description="Torch dtype hint ('auto', 'float16', 'bfloat16', etc.).",
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Whether to request an 8-bit load via bitsandbytes when available.",
    )

    class Config:
        env_prefix = "LLAMA_SERVER_"
        env_file = ".env"
        case_sensitive = False

    @validator("torch_dtype")
    def _normalize_dtype(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip().lower()
        return normalized if normalized else None


@lru_cache()
def get_settings() -> LlamaServerSettings:
    """Return cached environment-driven settings."""

    return LlamaServerSettings()


def expand_settings(
    base_settings: LlamaServerSettings, count: int
) -> List[LlamaServerSettings]:
    """
    Produce a list of settings objects for launching multiple servers.

    Each subsequent server reuses all configuration except the port, which is
    incremented sequentially starting from `base_settings.port`.
    """

    count = max(1, int(count))
    return [
        base_settings.model_copy(update={"port": base_settings.port + offset})
        for offset in range(count)
    ]
