#!/usr/bin/env python3
"""
Run a command and keep GPU utilization above a minimum by injecting short CUDA bursts
when utilization dips. Tested with PyTorch 2.x. Requires: pynvml, torch.

Example:
  python run_with_gpu_keepalive.py --gpu 0 --min-util 35 --check-interval 0.5 \
    --burst-ms 150 --matmul-dim 1536 --dtype fp16 -- \
    python your_nav_script.py --cfg config.yaml
"""
import argparse, os, subprocess, threading, time, math, shutil, sys
from datetime import datetime

# --------- Util readers (NVML preferred; fallback to nvidia-smi) ----------
def _has_pynvml():
    try:
        import pynvml  # noqa: F401
        return True
    except Exception:
        return False

class UtilReader:
    def __init__(self, gpu_index: int):
        self.gpu = gpu_index
        self.use_nvml = _has_pynvml()
        if self.use_nvml:
            import pynvml
            self.nvml = pynvml
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(self.gpu)
        else:
            if not shutil.which("nvidia-smi"):
                raise RuntimeError("Neither pynvml nor nvidia-smi available.")
    def read_percent(self) -> float:
        if self.use_nvml:
            s = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            return float(s.gpu)
        # fallback: parse nvidia-smi
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={self.gpu}", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        try:
            return float(out.decode("utf-8").strip().splitlines()[0])
        except Exception:
            return 0.0
    def close(self):
        if self.use_nvml:
            self.nvml.nvmlShutdown()

# --------- Keepalive worker (PyTorch) ----------
def keepalive_worker(gpu:int, min_util:float, check_interval:float,
                     burst_ms:int, matmul_dim:int, dtype:str,
                     stop_flag, verbose:bool):
    import torch
    device = torch.device(f"cuda:{gpu}")
    if dtype == "fp16":
        dt = torch.float16
    elif dtype == "bf16":
        dt = torch.bfloat16
    else:
        dt = torch.float32

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Pre-allocate small tensors to avoid frequent allocs
    with torch.cuda.device(device):
        A = torch.randn((matmul_dim, matmul_dim), device=device, dtype=dt)
        B = torch.randn((matmul_dim, matmul_dim), device=device, dtype=dt)

    reader = UtilReader(gpu)
    # Simple adaptive knob: scale number of matmuls based on gap to target
    while not stop_flag["stop"]:
        try:
            util = reader.read_percent()
        except Exception:
            util = 0.0

        if util < min_util:
            gap = max(0.0, min_util - util)  # e.g., 20 if util=10 and min=30
            # do k matmuls; k grows with gap but capped
            k = max(1, min(8, int(math.ceil(gap / 10.0))))
            t_end = time.perf_counter() + (burst_ms / 1000.0)
            with torch.cuda.device(device):
                for _ in range(k):
                    C = A @ B
                    A.add_(0.00001, C)  # tiny op to avoid DCE
                torch.cuda.synchronize()
            # ensure we don’t overshoot burst window too much
            while time.perf_counter() < t_end:
                time.sleep(0.001)

            if verbose:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] util={util:.1f}% < {min_util}%, "
                      f"burst k={k}, dim={matmul_dim}, dtype={dtype}", flush=True)
        else:
            time.sleep(check_interval)

    reader.close()

# --------- Main launcher ----------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gpu", type=int, default=0, help="GPU index to monitor/use")
    ap.add_argument("--min-util", type=float, default=35.0,
                    help="Minimum GPU utilization percent to maintain")
    ap.add_argument("--check-interval", type=float, default=0.5,
                    help="Seconds between utilization checks when above target")
    ap.add_argument("--burst-ms", type=int, default=150,
                    help="Approx burst compute duration when below target")
    ap.add_argument("--matmul-dim", type=int, default=1536,
                    help="Square GEMM dimension (trade utilization vs interference)")
    ap.add_argument("--dtype", choices=["fp16","bf16","fp32"], default="fp16",
                    help="Compute dtype for keepalive workload")
    ap.add_argument("--verbose", action="store_true", help="Print keepalive actions")
    ap.add_argument("--env-cuda-visible", action="store_true",
                    help="Also set CUDA_VISIBLE_DEVICES to the chosen --gpu")
    args, unknown = ap.parse_known_args()
    cmd = unknown

    if not args.cmd:
        print("ERROR: no command provided. Use: ... -- python your_script.py ...", file=sys.stderr)
        sys.exit(2)

    # Optionally restrict to the target GPU
    env = os.environ.copy()
    if args.env_cuda_visible:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Launch target command
    print(f"[launcher] Starting: {' '.join(args.cmd)}", flush=True)
    proc = subprocess.Popen(args.cmd, env=env)

    # Start keepalive thread
    stop_flag = {"stop": False}
    t = threading.Thread(
        target=keepalive_worker,
        args=(args.gpu, args.min_util, args.check_interval,
              args.burst_ms, args.matmul_dim, args.dtype, stop_flag, args.verbose),
        daemon=True,
    )
    t.start()

    # Wait for job to finish
    rc = 0
    try:
        rc = proc.wait()
    finally:
        stop_flag["stop"] = True
        t.join(timeout=5.0)
    print(f"[launcher] Exit code: {rc}", flush=True)
    sys.exit(rc)

if __name__ == "__main__":
    main()
