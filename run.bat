@echo off
rem Start the DR backend API on http://localhost:8000
cd /d "%~dp0"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
