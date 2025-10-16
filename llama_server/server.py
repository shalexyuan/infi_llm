import asyncio
import logging
import time
import uuid
from typing import Any, List, Literal, Optional, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from .config import get_settings, LlamaServerSettings
from .runtime import ChatTurn, LlamaRuntime

LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Meta LLaMA Chat Server",
    version="0.1.0",
    description="FastAPI server that hosts the meta-llam model.",
)

_SETTINGS_OVERRIDE_KEY = "_settings_override"


class TurnModel(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Role of the speaker generating the content.",
    )
    content: str = Field(..., description="Message content.")


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="Latest user prompt for the assistant.")
    history: Optional[List[TurnModel]] = Field(
        default=None,
        description="Previous conversation turns alternating between user and assistant.",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="Override for maximum tokens generated in this response.",
        ge=1,
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Override sampling temperature for this request.",
        ge=0.0,
        le=2.0,
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Override nucleus sampling top-p value.",
        ge=0.0,
        le=1.0,
    )

    @validator("history", each_item=True)
    def _strip_content(cls, turn: TurnModel) -> TurnModel:
        turn.content = turn.content.strip()
        return turn


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response text.")
    history: List[TurnModel] = Field(
        ...,
        description="Full conversation history including the latest turns.",
    )


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Any


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(
        default=None,
        description="Model identifier provided by the client (ignored).",
    )
    messages: Sequence[ChatCompletionMessage]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = Field(default=False)


class ChatCompletionChoice(BaseModel):
    index: int
    message: TurnModel
    finish_reason: Literal["stop"]


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    raise ValueError("Unsupported message content type.")


def _convert_messages(messages: Sequence[ChatCompletionMessage]) -> List[ChatTurn]:
    turns: List[ChatTurn] = []
    for message in messages:
        try:
            content_text = _normalize_content(message.content)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Unsupported message content payload.",
            ) from exc
        turns.append(ChatTurn(role=message.role, content=content_text))
    return turns


@app.on_event("startup")
async def load_runtime() -> None:
    """Load the heavy LLaMA runtime once the event loop is ready."""

    settings = getattr(app.state, _SETTINGS_OVERRIDE_KEY, None)
    if settings is None:
        settings = get_settings()
    else:
        # Preserve the override for other handlers (healthcheck, etc.).
        setattr(app.state, _SETTINGS_OVERRIDE_KEY, settings)
    LOGGER.info(
        "Initialising LLaMA server with model '%s' on %s:%s",
        settings.model_id,
        settings.host,
        settings.port,
    )
    runtime = await asyncio.to_thread(LlamaRuntime, settings)
    app.state.runtime = runtime
    app.state.settings = settings


@app.get("/healthz")
async def healthcheck() -> dict:
    """Simple liveness check."""

    runtime: Optional[LlamaRuntime] = getattr(app.state, "runtime", None)
    status = "ready" if runtime is not None else "initialising"
    return {"status": status}


@app.get("/v1/model_info")
async def model_info() -> dict:
    """Return metadata about the loaded model assets."""

    runtime: Optional[LlamaRuntime] = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    settings: LlamaServerSettings = app.state.settings
    return {
        "model_id": settings.model_id,
        "special_tokens": runtime.special_tokens_map,
        "max_input_tokens": settings.max_input_tokens,
        "default_generation": {
            "max_new_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
        },
    }


@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Generate a response from the LLaMA model."""

    runtime: Optional[LlamaRuntime] = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    history_turns = [
        ChatTurn(role=turn.role, content=turn.content)
        for turn in (payload.history or [])
    ]

    try:
        response_text = await asyncio.to_thread(
            runtime.generate,
            prompt=payload.prompt,
            history=history_turns,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    updated_history = list(payload.history or [])
    updated_history.append(TurnModel(role="user", content=payload.prompt.strip()))
    updated_history.append(TurnModel(role="assistant", content=response_text))

    return ChatResponse(response=response_text, history=updated_history)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions_endpoint(
    payload: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint."""

    if payload.stream:
        raise HTTPException(status_code=501, detail="Streaming not supported.")

    runtime: Optional[LlamaRuntime] = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    turns = _convert_messages(payload.messages)
    if not turns:
        raise HTTPException(status_code=400, detail="No messages provided.")

    prompt_turn = turns[-1]
    if prompt_turn.role != "user":
        raise HTTPException(
            status_code=400,
            detail="The final message must be from the user.",
        )

    history_turns = turns[:-1]

    try:
        response_text, usage = await asyncio.to_thread(
            runtime.generate,
            prompt=prompt_turn.content,
            history=history_turns,
            max_new_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            return_metadata=True,
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_message = TurnModel(role="assistant", content=response_text)
    choice = ChatCompletionChoice(
        index=0,
        message=response_message,
        finish_reason="stop",
    )

    settings: LlamaServerSettings = app.state.settings
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=settings.model_id,
        choices=[choice],
        usage=ChatCompletionUsage(**usage),
    )


def run(settings: Optional[LlamaServerSettings] = None) -> None:
    """Entry point used by scripts and `python -m llama_server.server`."""

    settings = settings or get_settings()
    setattr(app.state, _SETTINGS_OVERRIDE_KEY, settings)
    uvicorn.run(
        "llama_server.server:app",
        host=settings.host,
        port=settings.port,
        factory=False,
        reload=False,
    )


if __name__ == "__main__":
    run()
