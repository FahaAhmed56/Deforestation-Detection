@echo off
REM One-command setup for Windows: install -> data -> train -> evaluate -> demo -> GUI
cd /d %~dp0
python -m pip install -r requirements.txt
if not exist models\unet_forest.pt (
  if not exist data\processed\train python -m src.data.prepare_data
  python -m src.train
  python -m src.evaluate
)
if not exist outputs\runs\demo\run.json python -m src.pipeline --demo
python -m app.server
