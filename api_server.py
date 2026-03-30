"""
AURUM-OPT — Local API Server for Finals
Run: python api_server.py
Then open: http://localhost:8080
"""
import os, sys, json, threading, time, traceback, shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

BASE_DIR   = Path(__file__).resolve().parent
SRC_DIR    = BASE_DIR / "src"
STATIC_DIR = BASE_DIR          # dashboard.html is in project root
OUT_DIR    = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

app = Flask(__name__)
CORS(app)

# ── Job state ──────────────────────────────────────────────────────────────────
job = {"running":False,"cutoff":None,"csv":None,
       "status":"idle","message":"","started":None,"elapsed":None}
job_lock = threading.Lock()

def fmt_c(c): return str(c).rstrip("0").rstrip(".")

# ── Optimizer runner ───────────────────────────────────────────────────────────
def run_optimizer(csv_path: str, cutoff: float):
    with job_lock:
        job.update(running=True, status="running",
                   message=f"Loading {Path(csv_path).name}…",
                   started=time.time())
    try:
        import stope_optimizer as so
        so.main(csv_path=csv_path, cutoff=cutoff)
        elapsed = round(time.time() - job["started"], 1)

        # Read result written by optimizer
        cs = fmt_c(cutoff)
        payload = None
        for p in [BASE_DIR/f"results_{cs}gt.json", BASE_DIR/"results.json"]:
            if p.exists():
                payload = json.loads(p.read_text())
                break
        if not payload:
            raise RuntimeError("results.json not found after run")

        with job_lock:
            job.update(running=False, status="done", elapsed=elapsed,
                       message=f"Done {elapsed}s — {payload['stope_count']:,} stopes — {payload['gold_oz']/1e6:.3f}M oz")
    except Exception:
        tb = traceback.format_exc()
        print(f"[ERROR]\n{tb}", flush=True)
        with job_lock:
            job.update(running=False, status="error", message=tb)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    with job_lock:
        return jsonify(dict(job))

@app.route("/api/run", methods=["POST"])
def api_run():
    with job_lock:
        if job["running"]:
            return jsonify({"ok":False,"error":f"Already running {job['cutoff']} g/t"}), 409

    body = request.get_json(silent=True) or {}
    try:
        cutoff = float(body.get("cutoff", 10.0))
        assert 0.1 <= cutoff <= 50
    except:
        return jsonify({"ok":False,"error":"Invalid cutoff"}), 400

    csv_path = body.get("csv_path") or str(BASE_DIR / "Hackathon 2026 - Block Model.csv")
    if not Path(csv_path).exists():
        return jsonify({"ok":False,"error":f"CSV not found: {csv_path}"}), 400

    with job_lock:
        job.update(cutoff=cutoff, csv=csv_path, status="running",
                   message=f"Starting {cutoff} g/t…", elapsed=None)

    threading.Thread(target=run_optimizer, args=(csv_path, cutoff), daemon=True).start()
    return jsonify({"ok":True,"message":f"Started {cutoff} g/t"})

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accept a CSV block model upload, save it, return the path."""
    if "file" not in request.files:
        return jsonify({"ok":False,"error":"No file in request"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"ok":False,"error":"Must be a .csv file"}), 400

    save_path = BASE_DIR / f"uploaded_{f.filename}"
    f.save(str(save_path))
    print(f"[upload]  Saved: {save_path}", flush=True)
    return jsonify({"ok":True,"csv_path":str(save_path),"filename":f.filename})

@app.route("/api/dxf/<cs>")
def api_dxf(cs):
    for p in [OUT_DIR/f"stopes_optimised.dxf", OUT_DIR/"stopes_optimised.dxf"]:
        if p.exists():
            return send_file(str(p), as_attachment=True,
                             download_name=f"stopes_optimised_{cs}gt.dxf",
                             mimetype="application/octet-stream")
    return jsonify({"error":"DXF not found — run optimizer first"}), 404

@app.route("/api/csv_list")
def api_csv_list():
    """List available CSVs in project root."""
    csvs = [f.name for f in BASE_DIR.glob("*.csv")]
    return jsonify({"files": csvs})

@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "dashboard.html")

@app.route("/<path:filename>")
def static_files(filename):
    p = BASE_DIR / filename
    if p.exists():
        return send_from_directory(str(BASE_DIR), filename)
    return jsonify({"error": f"{filename} not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n{'='*60}")
    print(f"  AURUM-OPT  Local API Server")
    print(f"  http://localhost:{port}")
    print(f"  Base: {BASE_DIR}")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
