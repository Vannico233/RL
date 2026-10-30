"""Content identities for code and local model/adapter files, without secrets."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

from .data import sha256_file


def _identity(files, root):
    entries = [{"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
               for path in sorted(files)]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"sha256": digest, "files": entries}


def model_identity(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError("Use a downloaded local model/adapter directory")
    suffixes = {".safetensors", ".bin", ".json", ".model", ".txt", ".jinja"}
    files = [path for path in root.iterdir() if path.is_file() and path.suffix in suffixes]
    if not any(path.suffix in {".safetensors", ".bin"} for path in files):
        raise ValueError("No model or adapter weight files found")
    return _identity(files, root)


def code_identity(project):
    root = Path(project).resolve()
    files = [path for directory in ("src", "plugins", "scripts", "configs") for path in (root / directory).rglob("*")
             if path.is_file() and path.suffix in {".py", ".json", ".sh"}]
    files += [root / name for name in ("pyproject.toml", "requirements-server.txt", "constraints-server.txt", "environment.yml")]
    identity = _identity(files, root)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10)
        identity.update(git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
                        git_dirty=bool(status.stdout.strip()) if status.returncode == 0 else None)
    except (OSError, subprocess.TimeoutExpired):
        identity.update(git_commit=None, git_dirty=None)
    return identity


def package_versions():
    names = ("torch", "transformers", "peft", "vllm", "ms-swift", "trl", "accelerate", "datasets", "qwen-vl-utils", "Pillow")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "system": platform.system(), "packages": versions}
