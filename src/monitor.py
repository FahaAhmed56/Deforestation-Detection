"""Continuous monitoring: keep checking watched areas for new clearings.

Sentinel-2 revisits every location every ~5 days, so this script can run on
a schedule (cron, GitHub Actions, Windows Task Scheduler). For each area in
``watchlist.yaml`` it:

1. finds the newest clear Sentinel-2 scene since the last check,
2. compares it with the previous baseline scene,
3. writes a normal run (visible in the GUI) and an alert line if the cleared
   area is above the area's ``alert_ha`` threshold,
4. stores the new scene date as the next baseline.

    python -m src.monitor                       # check every area once
    # cron, every Monday 06:00:
    # 0 6 * * 1  cd /path/to/repo && python -m src.monitor >> monitor.log 2>&1
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import yaml

from src.config import ROOT, project_path
from src.pipeline import run_analysis, sentinel_scenes
from src.predict import ForestModel

WATCHLIST = ROOT / "watchlist.yaml"
STATE = ROOT / "outputs" / "monitor_state.json"


def main():
    areas = yaml.safe_load(WATCHLIST.read_text())["areas"]
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    model = ForestModel()
    alerts = []
    for area in areas:
        key = area["name"]
        baseline = state.get(key, area.get("baseline_date"))
        latest = date.today().isoformat()
        if baseline and (date.fromisoformat(latest) - date.fromisoformat(baseline)) < timedelta(days=10):
            print(f"[{key}] checked recently, skipping"); continue
        try:
            scenes = sentinel_scenes(tuple(area["bbox"]), [baseline, latest])
        except LookupError as e:
            print(f"[{key}] {e}"); continue
        if scenes[0]["date"] == scenes[1]["date"]:
            print(f"[{key}] no new scene yet"); continue
        meta = run_analysis(model, tuple(area["bbox"]), scenes, f"{key} (monitor)", "sentinel-2")
        loss = meta["totals"]["loss_ha"]
        if loss >= area.get("alert_ha", 1.0):
            alerts.append({"area": key, "from": scenes[0]["date"], "to": scenes[1]["date"],
                           "loss_ha": loss, "run": meta["id"]})
        state[key] = scenes[1]["date"]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))
    alert_file = project_path("runs_dir").parent / "alerts.jsonl"
    with open(alert_file, "a") as fh:
        for a in alerts:
            fh.write(json.dumps(a) + "\n")
            print(f"ALERT: {a['area']} lost {a['loss_ha']} ha between {a['from']} and {a['to']}")
    print(f"{len(alerts)} alert(s); state saved to {STATE}")


if __name__ == "__main__":
    main()
