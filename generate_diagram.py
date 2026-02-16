
import sys
import subprocess
from dotenv import load_dotenv
import os

# Load environment variables (Google API Key) from .env
load_dotenv()

if __name__ == "__main__":
    # Pass all arguments transparently to paperbanana
    # subprocess will inherit the current os.environ which has the loaded keys
    cmd = [sys.executable, "-m", "paperbanana.cli"] + sys.argv[1:]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(1)
