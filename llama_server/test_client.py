"""
Quick smoke-style test that proves `_query_llm_for_object` works with the
running llama_server. This script imports the helper from `main.py` and feeds
it a prompt via the `CogVLM2` client pointing at the FastAPI backend.

Run with:
    python -m llama_server.test_client --host 0.0.0.0 --port 31511
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from constants import LLM_OBJECT_SELECTION_SYSTEM_PROMPT
from src.vlm import CogVLM2


def _query_llm_for_object(llm_client: CogVLM2, prompt: str) -> str:
    if llm_client is None:
        return ""
    messages = [
        {
            "role": "system",
            "content": """You are a knowledgeable assistant. Answer with a single integer and nothing else""",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    try:
        _, content = llm_client.create_chat_completion(
            "idk",
            messages=messages,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            use_stream=False,
        )
        return content or ""
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[warning] LLM query failed: {exc}", file=sys.stderr)
        return ""


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invoke `_query_llm_for_object` against the llama server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server hostname.")
    parser.add_argument("--port", type=int, default=31511, help="Server port.")
    parser.add_argument(
        "--n-servers",
        type=int,
        default=1,
        help="Number of sequential ports to probe starting from --port.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            'Choose the ONE option the TV is most commonly found near in a home. '
            'Options: 1) "remote control", 2) "pillow", 3) "kettle". '
            "Reply with a single integer only."
        ),
        help="Prompt passed to the helper.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    prompt = args.prompt.strip()
    if not prompt:
        print("[error] Prompt must not be empty.", file=sys.stderr)
        return 2

    success_count = 0
    for offset in range(max(1, args.n_servers)):
        port = args.port + offset
        base_url = f"http://{args.host}:{port}"
        client = CogVLM2(base_url)
        print(f"\nTesting llama_server at {base_url}")
        print(f"Prompt: {prompt}")

        reply = (_query_llm_for_object(client, prompt) or "").strip()
        if not reply:
            print("[error] Helper returned an empty reply.", file=sys.stderr)
            continue

        if not reply.isdigit():
            print(f"[error] Expected numeric reply, got: {reply!r}", file=sys.stderr)
            continue

        print("Assistant reply:", reply)
        success_count += 1

    if success_count == 0:
        print("[error] All servers failed.", file=sys.stderr)
        return 5

    print(f"\nTest passed ✅ ({success_count}/{max(1, args.n_servers)} servers responded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
