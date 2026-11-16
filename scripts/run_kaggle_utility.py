"""Optional cached-head experiment; local +.02pp is not a robust advantage."""
from pathlib import Path
import subprocess
import sys

bundle = next(Path("/kaggle/input").rglob("tacsx_utility.pyz"))
subprocess.run([sys.executable, str(bundle), "--output", "/kaggle/working/tacsx_utility_run"], check=True)
