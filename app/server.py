"""ForestWatch web server (FastAPI).

    python -m app.server            # → http://localhost:8000

REST API
    GET  /api/health                model + version info
    GET  /api/model                 training history + test metrics
    GET  /api/runs                  list of analyses
    GET  /api/runs/{id}             one analysis (run.json)
    DELETE /api/runs/{id}           delete an analysis
    POST /api/analyze               start a new analysis (background job)
    GET  /api/jobs/{id}             job status + live log
    GET  /runs/{id}/{file}          overlay PNGs, GeoJSON, figures
"""
from __future__ import annotations

import json
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src import __version__
from src.config import ROOT, load_config, project_path

STATIC = Path(__file__).parent / "static"
RUNS = project_path("runs_dir")

app = FastAPI(title="ForestWatch API", version=__version__)
executor = ThreadPoolExecutor(max_workers=1)        # one analysis at a time (CPU bound)
jobs: dict[str, dict] = {}
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from src.predict import ForestModel
            _model = ForestModel()
        return _model


class AnalyzeRequest(BaseModel):
    source: str = Field("sentinel", pattern="^(sentinel|demo)$")
    name: str = "Custom area"
    bbox: list[float] | None = None          # [west, south, east, north]
    dates: list[str] = []

    @field_validator("dates")
    @classmethod
    def _dates(cls, v):
        for d in v:
            date.fromisoformat(d)
        return sorted(v)


def _job(job_id: str, req: AnalyzeRequest):
    job = jobs[job_id]

    def log(msg):
        job["log"].append(str(msg))

    try:
        from src.pipeline import DEMO, demo_scenes, run_analysis, sentinel_scenes
        job["status"] = "running"
        model = get_model()
        if req.source == "demo":
            log("Generating demo scenes …")
            bbox, scenes, truth = demo_scenes()
            meta = run_analysis(model, bbox, scenes, DEMO["name"], "demo", truth, log, run_id="demo")
        else:
            bbox = tuple(req.bbox)
            meta = run_analysis(model, bbox, sentinel_scenes(bbox, req.dates, log), req.name, "sentinel-2", None, log)
        job.update(status="done", run_id=meta["id"])
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        job.update(status="error", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc(limit=2))


@app.get("/api/health")
def health():
    ck = project_path("models_dir") / "unet_forest.pt"
    return {"status": "ok", "version": __version__, "model_trained": ck.exists()}


@app.get("/api/model")
def model_info():
    md = project_path("models_dir")
    out = {}
    for name in ("history", "test_metrics"):
        f = md / f"{name}.json"
        out[name] = json.loads(f.read_text()) if f.exists() else None
    out["config"] = load_config()["model"]
    return out


@app.get("/api/runs")
def list_runs():
    runs = []
    for f in sorted(RUNS.glob("*/run.json")):
        m = json.loads(f.read_text())
        runs.append({k: m.get(k) for k in ("id", "name", "source", "created", "totals")}
                    | {"dates": [s["date"] for s in m["scenes"]]})
    return sorted(runs, key=lambda r: r["created"], reverse=True)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    f = RUNS / run_id / "run.json"
    if not f.resolve().is_relative_to(RUNS.resolve()) or not f.exists():
        raise HTTPException(404, "run not found")
    return json.loads(f.read_text())


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    d = (RUNS / run_id).resolve()
    if not d.is_relative_to(RUNS.resolve()) or not d.exists():
        raise HTTPException(404, "run not found")
    shutil.rmtree(d)
    return {"deleted": run_id}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if req.source == "sentinel":
        if not req.bbox or len(req.bbox) != 4:
            raise HTTPException(422, "draw an area of interest (bbox) first")
        if len(req.dates) < 2:
            raise HTTPException(422, "pick at least two dates")
    job_id = uuid.uuid4().hex[:10]
    jobs[job_id] = {"id": job_id, "status": "queued", "log": [], "run_id": None, "error": None}
    executor.submit(_job, job_id, req)
    return jobs[job_id]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "job not found")
    return jobs[job_id]


app.mount("/runs", StaticFiles(directory=RUNS), name="runs")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/figures", StaticFiles(directory=project_path("images_dir")), name="figures")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


def main():
    import uvicorn
    s = load_config()["server"]
    print(f"ForestWatch → http://localhost:{s['port']}")
    uvicorn.run(app, host=s["host"], port=s["port"])


if __name__ == "__main__":
    main()
