#!/usr/bin/env python3
"""Validate the candidate Linux/CUDA environment before starting either process."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from chart_rl.framework_source import verify_framework_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/environment/preflight.json")
    parser.add_argument("--gpu-count", type=int, default=2)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs/project.json").read_text(encoding="utf-8"))
    checks = {"linux": platform.system() == "Linux", "python_3_11": sys.version_info[:2] == (3, 11)}
    versions = {}
    for line in (project / "constraints-server.txt").read_text().splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        name, expected = line.strip().split("==")
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        checks["version_" + name] = versions[name] is not None and versions[name].split("+")[0] == expected
    direct = {}
    try:
        swift = importlib.metadata.distribution("ms-swift")
        versions["ms-swift"] = swift.version
        direct = json.loads(swift.read_text("direct_url.json") or "{}")
    except importlib.metadata.PackageNotFoundError:
        versions["ms-swift"] = None
    source = verify_framework_source(direct, config)
    checks["framework_version"] = versions["ms-swift"] == config["framework_version"]
    checks["framework_commit"] = source["verified"]
    gpu_info, cuda_error, runtime = [], None, None
    if versions.get("torch") and checks["linux"]:
        try:
            import torch
            runtime = torch.version.cuda
            checks["cuda_available"] = torch.cuda.is_available()
            checks["gpu_count"] = torch.cuda.device_count() >= args.gpu_count
            for index in range(min(torch.cuda.device_count(), args.gpu_count)):
                with torch.cuda.device(index):
                    properties = torch.cuda.get_device_properties(index)
                    free, total = torch.cuda.mem_get_info()
                    tensor = torch.ones((64, 64), device=f"cuda:{index}", dtype=torch.bfloat16)
                    result = tensor @ tensor
                    torch.cuda.synchronize()
                    kernel_ok = bool((result == 64).all().item())
                    gpu_info.append({"index": index, "name": properties.name,
                                     "total_GiB": round(total / 1024 ** 3, 2), "free_GiB": round(free / 1024 ** 3, 2),
                                     "capability": [properties.major, properties.minor], "bfloat16_matmul_ok": kernel_ok})
                    checks[f"gpu_{index}_kernel"] = kernel_ok
                    del result, tensor
            # Import the exact plugin/API without constructing a training engine.
            import runpy
            runpy.run_path(str(project / "plugins/chartqa_plugin.py"))
            checks["plugin_import"] = True
        except Exception as error:
            cuda_error = f"{type(error).__name__}: {error}"
            checks["gpu_and_plugin_initialization"] = False
    else:
        checks["gpu_and_plugin_initialization"] = False
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120)
    checks["pip_check"] = pip_check.returncode == 0
    report = {"status": "pass" if all(checks.values()) else "needs_attention", "checks": checks,
              "python": platform.python_version(), "system": platform.platform(), "versions": versions,
              "framework_commit": source["commit"], "framework_source": source,
              "torch_cuda_runtime": runtime, "gpus": gpu_info, "error": cuda_error,
              "pip_check": pip_check.stdout.strip(),
              "note": "Passing environment checks does not prove VLM rollout, GRPO update, or experimental gains."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
