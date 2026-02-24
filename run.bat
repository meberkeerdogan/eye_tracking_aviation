@echo off
REM ── Eye Tracking Aviation launcher ────────────────────────────────────────
REM Runs the app using the project's own virtual environment (PySide6 6.9.x).
REM Double-click this file, or run it from any terminal.

cd /d "%~dp0"
.venv\Scripts\python app\main.py
