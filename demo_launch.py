"""Forward npm-style host/port arguments to the project Python and Streamlit."""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    # Kimi Work may invoke npm with its managed Python; re-exec the project env
    # when Streamlit is available there. ASTROCYTE_PYTHON supports other setups.
    if importlib.util.find_spec("streamlit") is None:
        candidates = [os.environ.get("ASTROCYTE_PYTHON"), str(Path.home() / ".conda/envs/astrocyte/python.exe")]
        for candidate in candidates:
            if candidate and Path(candidate).is_file() and Path(candidate).resolve() != Path(sys.executable).resolve():
                return subprocess.call([candidate, str(Path(__file__).resolve()), *sys.argv[1:]])
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7100)
    args, _ = parser.parse_known_args()
    return subprocess.call([
        sys.executable, "-m", "streamlit", "run", "demo_app.py",
        "--server.address", args.host, "--server.port", str(args.port),
        "--server.headless", "true",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
