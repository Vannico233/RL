#!/usr/bin/env python3
"""Run CPU contract tests without installing training dependencies."""

from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import tempfile
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", help="Optional JSON evidence report from this actual test execution")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    # Keep all generated fixtures inside this checkout, including on Windows.
    temporary_root = project / ".test-tmp"
    temporary_root.mkdir(exist_ok=True)
    tempfile.tempdir = str(temporary_root)
    suite = unittest.defaultTestLoader.discover(str(project / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.report:
        import PIL
        source_files = [path for directory in ("src", "scripts", "plugins", "tests", "configs")
                        for path in (project / directory).rglob("*") if path.suffix in (".py", ".json")]
        hashes = {path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_files)}
        report = {"status": "pass" if result.wasSuccessful() else "fail", "tests_run": result.testsRun,
                  "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                  "system": platform.system(), "pillow": PIL.__version__, "source_sha256": hashes,
                  "validation_scope": "CPU project logic and an explicit ms-swift scheduler contract double",
                  "gpu_model_or_training_executed": False}
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
