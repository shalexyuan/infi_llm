import argparse
from multiprocessing import Process
from typing import Dict, List, Optional

from .config import LlamaServerSettings, expand_settings, get_settings
from .server import run


def _build_settings(overrides: Dict[str, Optional[object]]) -> LlamaServerSettings:
    base = get_settings().model_dump()
    base.update({k: v for k, v in overrides.items() if v is not None})
    return LlamaServerSettings(**base)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Meta LLaMA-2 FastAPI server.",
    )
    parser.add_argument("--host", help="Override bind host (default 0.0.0.0).")
    parser.add_argument(
        "--port",
        type=int,
        help="Override bind port (default 8000).",
    )
    parser.add_argument(
        "--model-id",
        help="Hugging Face model ID to load (default meta-llama/Llama-2-7b-hf).",
    )
    parser.add_argument(
        "--hf-token",
        help="Explicit Hugging Face token to use for gated models.",
    )
    parser.add_argument(
        "--n-servers",
        type=int,
        default=1,
        help=(
            "Number of server instances to launch. Each additional server "
            "binds to the next sequential port."
        ),
    )
    args = parser.parse_args()

    settings = _build_settings(
        {
            "host": args.host,
            "port": args.port,
            "model_id": args.model_id,
            "hf_token": args.hf_token,
        }
    )
    server_settings: List[LlamaServerSettings] = expand_settings(
        settings, args.n_servers or 1
    )

    if len(server_settings) == 1:
        run(settings=server_settings[0])
        return

    processes: List[Process] = []
    try:
        for child_settings in server_settings:
            proc = Process(
                target=run,
                kwargs={"settings": child_settings},
                daemon=False,
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.join()


if __name__ == "__main__":
    main()
