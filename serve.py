"""
AURUM-OPT — Finals Startup Script
Run this ONE script to start everything:
    python serve.py

Then open: http://localhost:8080
"""
import sys, subprocess, os
from pathlib import Path

BASE = Path(__file__).resolve().parent

# Check flask is installed
try:
    import flask
    import flask_cors
except ImportError:
    print("[setup]  Installing flask and flask-cors...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "flask", "flask-cors", "-q"])
    print("[setup]  Done.")

print("""
╔══════════════════════════════════════════════════╗
║  AURUM-OPT  v2.0  —  Finals Mode                ║
║  http://localhost:8080                           ║
║                                                  ║
║  1. Open http://localhost:8080 in Chrome         ║
║  2. Upload new block model CSV (if needed)       ║
║  3. Type cutoff grade and click Run ▶            ║
║  4. Results appear in ~30 seconds                ║
║  5. Export DXF when ready                        ║
╚══════════════════════════════════════════════════╝
""")

# Run the API server
os.chdir(BASE)
from api_server import app
port = int(os.environ.get("PORT", 8080))
app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
