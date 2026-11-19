@echo off
rem TACS-X demo with image upload: starts the local model server and opens the page.
rem Close this window to stop the server. Needs .venv-gpu (CUDA) and the outputs/cub200_selftrain_v1 checkpoints.
cd /d "%~dp0"
title TACS-X demo server
echo Starting the TACS-X demo (about 1 minute). Keep this window open; close it to stop.
".venv-gpu\Scripts\python.exe" scripts\serve_demo.py --open
if errorlevel 1 pause
