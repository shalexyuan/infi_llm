"""
Quick smoke test for the LLaMA chat server that mirrors the navigation prompt.

This script sends a short chat-completions style request using the same system
prompt as `_query_llm_for_object` in `main.py` and prints the assistant reply
along with token usage. It exits with a non-zero status if the server is
unreachable or returns an error response.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

import requests

from constants import LLM_OBJECT_SELECTION_SYSTEM_PROMPT


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a test request to the local LLaMA server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server hostname.")
    parser.add_argument("--port", type=int, default=31511, help="Server port.")
    parser.add_argument(
        "--prompt",
        default=(
            'Choose the ONE option the TV is most commonly found near in a home. '
            'Options: 1) "remote control", 2) "pillow", 3) "kettle". '
            "Reply with a single integer only."
        ),
        help="User prompt used for the test request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature override for the request.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling threshold override.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    endpoint = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = {
        "model": "meta-llama/test",
        "messages": [
            {"role": "system", "content": """You are a knowledgeable assistant. Answer with a single integer and nothing else"""},
            {"role": "user", "content": """Choose the ONE option the TV is most commonly found near in a home. Options: 1) "remote control", 2) "pillow", 3) "kettle". Reply with a single integer only."""},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stream": False,
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=180)
    except requests.RequestException as exc:
        print(f"[error] Failed to reach server: {exc}", file=sys.stderr)
        return 2
    print("response: ", response)
    if response.status_code != 200:
        print(
            f"[error] Server returned {response.status_code}: {response.text}",
            file=sys.stderr,
        )
        return 3

    body = response.json()
    print(json.dumps(body, indent=2))

    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {}).get("content", "")
    usage = body.get("usage", {})

    print("\nAssistant reply:", message)
    if usage:
        print("Token usage:", usage)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
