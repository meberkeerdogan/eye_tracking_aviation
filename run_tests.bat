@echo off
REM ── Run all tests ──────────────────────────────────────────────────────────
cd /d "%~dp0"
.venv\Scripts\pytest tests\ -v
pause
