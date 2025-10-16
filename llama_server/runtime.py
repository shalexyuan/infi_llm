import logging
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Iterable, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from .config import LlamaServerSettings

LOGGER = logging.getLogger(__name__)


@dataclass
class ChatTurn:
    """Represents a single conversational turn."""

    role: str
    content: str


class LlamaRuntime:
    """
    Load and serve a Meta LLaMA-2 causal LM with conversation helpers.

    This class performs all heavyweight loading (config, tokenizer, model,
    generation config) up front so the FastAPI layer can reuse the instance.
    """

    def __init__(self, settings: LlamaServerSettings) -> None:
        self._settings = settings
        self._lock = Lock()

        LOGGER.info("Loading config for %s", settings.model_id)
        self.config = AutoConfig.from_pretrained(
            settings.model_id,
            token=settings.hf_token,
            trust_remote_code=False,
        )

        LOGGER.info("Loading tokenizer for %s", settings.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_id,
            token=settings.hf_token,
            padding_side="left",
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            # Ensure a pad token exists; reuse EOS when missing.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        LOGGER.info("Loading generation config for %s", settings.model_id)
        try:
            self.generation_config = GenerationConfig.from_pretrained(
                settings.model_id,
                token=settings.hf_token,
            )
        except Exception:
            LOGGER.warning(
                "No pre-defined generation config found; using defaults."
            )
            self.generation_config = GenerationConfig.from_model_config(
                self.config
            )

        load_kwargs: Dict[str, object] = {
            "device_map": settings.device_map,
            "torch_dtype": self._resolve_dtype(settings.torch_dtype),
        }
        if settings.load_in_8bit:
            load_kwargs["load_in_8bit"] = True

        LOGGER.info("Loading model weights for %s", settings.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            config=self.config,
            token=settings.hf_token,
            **load_kwargs,
        )
        self.model.eval()

        # Expose special tokens for downstream consumers if needed.
        self.special_tokens_map = self.tokenizer.special_tokens_map

        self._target_device = self._infer_device_obj()

        LOGGER.info(
            "LLaMA runtime ready (device=%s)",
            self._target_device if self._target_device else "auto",
        )

    def _resolve_dtype(self, dtype_hint: Optional[str]) -> Optional[torch.dtype]:
        """Translate dtype strings to torch dtypes when possible."""

        if dtype_hint in (None, "", "auto"):
            return None
        lookup = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
        }
        key = dtype_hint.lower()
        if key not in lookup:
            LOGGER.warning("Unsupported torch dtype hint '%s'; using auto.", dtype_hint)
            return None
        return lookup[key]

    def _infer_device_obj(self) -> Optional[torch.device]:
        """Return a torch.device for weight placement when possible."""

        device_attr = getattr(self.model, "device", None)
        if isinstance(device_attr, torch.device) and device_attr.type != "meta":
            return device_attr
        if isinstance(device_attr, str) and device_attr != "meta":
            return torch.device(device_attr)

        try:
            first_param = next(self.model.parameters())
            if first_param.device.type != "meta":
                return first_param.device
        except StopIteration:
            return None
        return None

    def build_prompt(self, history: Iterable[ChatTurn], prompt: str) -> str:
        """
        Format the conversation into a single prompt according to the LLaMA-2
        chat template when available.
        """

        chat_template = getattr(self.tokenizer, "chat_template", None)
        messages: List[Dict[str, str]] = [
            {"role": turn.role, "content": turn.content} for turn in history
        ]
        messages.append({"role": "user", "content": prompt})

        if chat_template:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            return rendered

        # Fallback: simple role-prefixed format.
        buffer: List[str] = []
        for turn in messages[:-1]:
            buffer.append(f"{turn['role'].capitalize()}: {turn['content']}")
        buffer.append(f"User: {messages[-1]['content']}")
        buffer.append("Assistant:")
        return "\n".join(buffer)

    def generate(
        self,
        prompt: str,
        history: Optional[Iterable[ChatTurn]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        return_metadata: bool = False,
    ) -> str | tuple[str, Dict[str, int]]:
        """
        Run text generation for the provided prompt and optional history.
        """

        history = list(history or [])
        full_prompt = self.build_prompt(history, prompt)
        encoded = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._settings.max_input_tokens,
        )

        if "token_type_ids" in encoded:
            encoded.pop("token_type_ids")

        if self._target_device:
            encoded = {k: v.to(self._target_device) for k, v in encoded.items()}
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or self._settings.max_new_tokens,
            temperature=temperature or self._settings.temperature,
            top_p=top_p or self._settings.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with self._lock:
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    **gen_kwargs,
                )

        # Only return the newly generated portion.
        prompt_length = encoded["input_ids"].shape[-1]
        completion_tokens = generated[0].shape[-1] - prompt_length
        generated_text = self.tokenizer.decode(
            generated[0][prompt_length:],
            skip_special_tokens=True,
        )
        text = generated_text.strip()
        if not return_metadata:
            return text

        usage = {
            "prompt_tokens": int(prompt_length),
            "completion_tokens": int(max(completion_tokens, 0)),
            "total_tokens": int(prompt_length + max(completion_tokens, 0)),
        }
        return text, usage
