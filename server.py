"""
AURUM-OPT — Flask API Server for Render.com
Simple approach: let optimizer write wherever it wants,
then find and copy the results into DATA_DIR.
"""

import os, sys, json, threading, time, traceback, shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
SRC_DIR    = BASE_DIR / "src"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR   = BASE_DIR / "data"
OUT_DIR    = BASE_DIR / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

CSV_PATH = BASE_DIR / "Hackathon 2026 - Block Model.csv"

sys.path.insert(0, str(SRC_DIR))

app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

# ── Job state ─────────────────────────────────────────────────────────────────
job = {
    "running": False,
    "cutoff":  None,
    "status":  "idle",
    "message": "",
    "started": None,
    "elapsed": None,
}
job_lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_cutoff(c):
    return str(c).rstrip("0").rstrip(".")

def update_index(cutoff):
    idx_file = DATA_DIR / "results_index.json"
    cutoffs = []
    if idx_file.exists():
        try:
            cutoffs = json.loads(idx_file.read_text()).get("cutoffs", [])
        except Exception:
            pass
    if cutoff not in cutoffs:
        cutoffs.append(cutoff)
        cutoffs.sort()
    idx_file.write_text(json.dumps({"cutoffs": cutoffs}, indent=2))

# ── Optimizer runner ──────────────────────────────────────────────────────────
def run_optimizer(cutoff: float):
    with job_lock:
        job["running"] = True
        job["status"]  = "running"
        job["message"] = f"Running optimizer for {cutoff} g/t…"
        job["started"] = time.time()

    try:
        import stope_optimizer as so
        cs = fmt_cutoff(cutoff)

        # Run the optimizer — it writes results.json and results_Xgt.json
        # to BASE_DIR (the project root, which is SRC_DIR/..)
        so.main(csv_path=str(CSV_PATH), cutoff=cutoff)

        # Search the entire project for any results JSON it produced
        # The optimizer uses _get_output_dir which resolves to BASE_DIR
        search_dirs = [BASE_DIR, SRC_DIR, SRC_DIR.parent, OUT_DIR, DATA_DIR]
        payload     = None
        found_at    = None

        # First look for the cutoff-specific file
        for d in search_dirs:
            p = d / f"results_{cs}gt.json"
            if p.exists():
                try:
                    payload  = json.loads(p.read_text())
                    found_at = p
                    break
                except Exception:
                    continue

        # Fall back to results.json
        if payload is None:
            for d in search_dirs:
                p = d / "results.json"
                if p.exists():
                    try:
                        payload  = json.loads(p.read_text())
                        found_at = p
                        break
                    except Exception:
                        continue

        if payload is None:
            # Log everything we can find for debugging
            all_found = list(BASE_DIR.rglob("results*.json"))
            raise RuntimeError(
                f"Optimizer ran but no results JSON found anywhere.\n"
                f"Searched dirs: {[str(d) for d in search_dirs]}\n"
                f"All results files in project: {[str(f) for f in all_found]}"
            )

        print(f"[server]  Found results at: {found_at}", flush=True)

        # Copy into DATA_DIR — this is what the dashboard reads
        cs_dest = DATA_DIR / f"results_{cs}gt.json"
        latest  = DATA_DIR / "results.json"
        content = json.dumps(payload, indent=2)
        cs_dest.write_text(content)
        latest.write_text(content)
        print(f"[server]  → {cs_dest}", flush=True)
        print(f"[server]  → {latest}", flush=True)

        # Copy DXF
        for dxf_src in [OUT_DIR / "stopes_optimised.dxf",
                         BASE_DIR / "stopes_optimised.dxf"]:
            if dxf_src.exists():
                shutil.copy2(str(dxf_src), str(DATA_DIR / f"stopes_optimised_{cs}gt.dxf"))
                break

        update_index(cutoff)

        elapsed = round(time.time() - job["started"], 1)
        with job_lock:
            job["running"] = False
            job["status"]  = "done"
            job["elapsed"] = elapsed
            job["message"] = (
                f"Done in {elapsed}s — "
                f"{payload['stope_count']:,} stopes — "
                f"{payload['gold_oz']/1e6:.3f}M oz"
            )

    except Exception:
        tb = traceback.format_exc()
        print(f"[ERROR]\n{tb}", flush=True)
        with job_lock:
            job["running"] = False
            job["status"]  = "error"
            job["message"] = tb

# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    with job_lock:
        return jsonify({
            "status":  job["status"],
            "message": job["message"],
            "cutoff":  job["cutoff"],
            "elapsed": job["elapsed"],
        })


@app.route("/api/run", methods=["POST"])
def api_run():
    with job_lock:
        if job["running"]:
            return jsonify({
                "ok": False,
                "error": f"Already running {job['cutoff']} g/t — please wait"
            }), 409

    body = request.get_json(silent=True) or {}
    try:
        cutoff = float(body.get("cutoff", 10.0))
        if not (0.1 <= cutoff <= 30):
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid cutoff value"}), 400

    if not CSV_PATH.exists():
        return jsonify({
            "ok": False,
            "error": f"Block model CSV not found: {CSV_PATH.name}"
        }), 500

    with job_lock:
        job["cutoff"]  = cutoff
        job["status"]  = "running"
        job["message"] = f"Starting {cutoff} g/t…"
        job["elapsed"] = None

    threading.Thread(target=run_optimizer, args=(cutoff,), daemon=True).start()
    return jsonify({"ok": True, "message": f"Started {cutoff} g/t"})


@app.route("/api/dxf/<cs>")
def api_dxf(cs):
    for p in [DATA_DIR / f"stopes_optimised_{cs}gt.dxf",
               OUT_DIR  / "stopes_optimised.dxf"]:
        if p.exists():
            return send_file(str(p), as_attachment=True,
                             download_name=f"stopes_optimised_{cs}gt.dxf",
                             mimetype="application/octet-stream")
    return jsonify({"error": "DXF not found"}), 404


# ══════════════════════════════════════════════════════════════════════════════
# FILE SERVING — all JSON requests come here
# The dashboard fetches: results.json, results_Xgt.json, results_index.json
# All served from DATA_DIR
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "dashboard.html")


@app.route("/<path:filename>")
def catch_all(filename):
    # 1. Static files (dashboard.html, points3d.json, etc.)
    p = STATIC_DIR / filename
    if p.exists():
        return send_from_directory(str(STATIC_DIR), filename)

    # 2. Everything else comes from DATA_DIR
    #    This covers results.json, results_10gt.json, results_index.json
    p = DATA_DIR / filename
    if p.exists():
        return send_from_directory(str(DATA_DIR), filename)

    return jsonify({"error": f"{filename} not found"}), 404


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"[aurum-opt]  port={port}")
    print(f"[aurum-opt]  csv={CSV_PATH}  exists={CSV_PATH.exists()}")
    print(f"[aurum-opt]  data={DATA_DIR}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
