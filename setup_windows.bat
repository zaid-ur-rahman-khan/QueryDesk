@echo off
setlocal
where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher not found. Install Python 3.12 x64 from python.org.
  pause
  exit /b 1
)
py -3.12 --version
if errorlevel 1 (
  echo Python 3.12 is not installed. Install Python 3.12 x64, then run this file again.
  pause
  exit /b 1
)
if not exist .venv (
  py -3.12 -m venv .venv
  if errorlevel 1 exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
if not exist .env copy .env.example .env
echo.
echo Setup complete. Add GOOGLE_API_KEY to .env, then run:
echo   .venv\Scripts\activate
echo   streamlit run agent.py
pause
