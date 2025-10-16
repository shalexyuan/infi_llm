import argparse
import sys
from typing import List

import requests

from .server import TurnModel


def _print_banner(host: str, port: int) -> None:
    print(f"Connected to LLaMA chat server at http://{host}:{port}")
    print("Type '/reset' to clear the conversation or '/exit' to quit.\n")


def _send_message(
    endpoint: str,
    prompt: str,
    history: List[TurnModel],
) -> List[TurnModel]:
    payload = {
        "prompt": prompt,
        "history": [turn.dict() for turn in history],
    }
    response = requests.post(endpoint, json=payload, timeout=600)
    response.raise_for_status()
    body = response.json()
    turns = [TurnModel(**turn) for turn in body["history"]]
    print(f"LLaMA: {body['response']}\n")
    return turns


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simple terminal chatbot client for the LLaMA server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server hostname.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--system",
        help="Optional system prompt that seeds the conversation.",
    )
    args = parser.parse_args(argv)

    endpoint = f"http://{args.host}:{args.port}/v1/chat"
    history: List[TurnModel] = []

    if args.system:
        history.append(TurnModel(role="system", content=args.system))

    _print_banner(args.host, args.port)

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not prompt:
            continue
        if prompt.lower() in ("/exit", "quit", "exit"):
            print("Goodbye.")
            return 0
        if prompt.lower() == "/reset":
            history = []
            if args.system:
                history.append(TurnModel(role="system", content=args.system))
            print("Conversation cleared.\n")
            continue

        try:
            history = _send_message(endpoint, prompt, history)
        except requests.HTTPError as exc:
            print(f"Server error: {exc.response.text}", file=sys.stderr)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
