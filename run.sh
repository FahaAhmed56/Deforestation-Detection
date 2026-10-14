#!/usr/bin/env bash
# One-command setup: install → data → train → evaluate → demo → start GUI.
# Skips steps whose outputs already exist (delete models/unet_forest.pt to retrain).
set -e
cd "$(dirname "$0")"

python -m pip install -r requirements.txt

if [ ! -f models/unet_forest.pt ]; then
  [ -d data/processed/train ] || python -m src.data.prepare_data
  python -m src.train
  python -m src.evaluate
fi
[ -f outputs/runs/demo/run.json ] || python -m src.pipeline --demo

python -m app.server
