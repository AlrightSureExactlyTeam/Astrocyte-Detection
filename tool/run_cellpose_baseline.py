import sys
from real_pipeline_cli import main

if __name__ == "__main__":
    sys.argv.extend(["--model", "cellpose"])
    raise SystemExit(main("predict"))
