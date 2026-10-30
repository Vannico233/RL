#!/usr/bin/env python3
"""Read-only server report for Chart Active Vision RL (Python 3.8+).

Run in the Python/Conda environment intended for training:
    python3 server_env_probe.py > server_env.json
    python3 server_env_probe.py --disk-path /data > server_env.json

Uses the standard library only. Does not install packages, download models,
connect to the network, or read credentials/environment variables. Package
versions are read without importing CUDA libraries. Inspect the JSON before
sharing it; disk paths and installed package versions are included.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


PACKAGES = (
    "torch", "torchvision", "torchaudio", "transformers", "accelerate",
    "peft", "trl", "vllm", "ms-swift", "deepspeed", "qwen-vl-utils",
    "datasets", "huggingface-hub", "flash-attn", "safetensors",
)


def command_report(command, timeout=10):
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False}
    try:
        result = subprocess.run(
            [executable] + command[1:],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace").strip(),
            "stderr": result.stderr.decode("utf-8", errors="replace").strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "error": str(error)}


def linux_os_release():
    path = Path("/etc/os-release")
    if not path.is_file():
        return None
    values = {}
    allowed = {"ID", "ID_LIKE", "VERSION_ID", "PRETTY_NAME"}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key in allowed:
                values[key] = value.strip().strip('"')
        return values
    except OSError as error:
        return {"error": str(error)}


def memory_report():
    path = Path("/proc/meminfo")
    if not path.is_file():
        return {"available": False, "note": "Linux /proc/meminfo unavailable"}
    try:
        result = {"available": True}
        keys = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
        for line in path.read_text(encoding="ascii").splitlines():
            key, _, value = line.partition(":")
            if key in keys:
                result[key + "_GiB"] = round(int(value.split()[0]) / 1024 ** 2, 2)
        return result
    except (OSError, ValueError, IndexError) as error:
        return {"available": False, "error": str(error)}


def disk_report(path):
    resolved = Path(path).expanduser().resolve()
    try:
        usage = shutil.disk_usage(str(resolved))
        return {
            "path": str(resolved),
            "total_GiB": round(usage.total / 1024 ** 3, 2),
            "free_GiB": round(usage.free / 1024 ** 3, 2),
        }
    except OSError as error:
        return {"path": str(resolved), "error": str(error)}


def package_report():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        except Exception as error:
            versions[name] = {"error": str(error)}
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disk-path", action="append", default=[],
        help="Existing model/data/output directory to inspect; may be repeated.",
    )
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "purpose": "Chart Active Vision RL server compatibility preflight",
        "system": {
            "os": platform.system(),
            "kernel": platform.release(),
            "architecture": platform.machine(),
            "distribution": linux_os_release(),
            "logical_cpu_count": os.cpu_count(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "memory": memory_report(),
        "disks": [disk_report(path) for path in (args.disk_path or [Path.cwd()])],
        "gpu_inventory": command_report([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]),
        "gpu_inventory_columns": ["name", "memory_total_MiB", "driver_version"],
        "nvidia_versions": command_report(["nvidia-smi", "--version"]),
        "cuda_compiler": command_report(["nvcc", "--version"]),
        "conda": command_report(["conda", "--version"]),
        "docker": command_report(["docker", "--version"]),
        "git": command_report(["git", "--version"]),
        "packages": package_report(),
        "notes": [
            "GPU inventory is host-visible hardware, not proof of GPU allocation.",
            "Package versions describe the Python interpreter running this script.",
            "CUDA runtime compatibility and training have not been tested.",
            "Network access, model paths, and dataset paths need separate confirmation.",
            "Memory and disk availability are point-in-time observations.",
        ],
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
