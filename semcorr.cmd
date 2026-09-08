@echo off
setlocal EnableExtensions

set "SEMCORR_ROOT=%~dp0"
set "SEMCORR_VENV=%SEMCORR_ROOT%.venv"
set "SEMCORR_VENV_PY=%SEMCORR_VENV%\Scripts\python.exe"

if defined SEMCORR_PYTHON (
    set "SEMCORR_BOOTSTRAP_PY=%SEMCORR_PYTHON%"
) else (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "SEMCORR_BOOTSTRAP_PY=%%P"
    if not defined SEMCORR_BOOTSTRAP_PY (
        for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "SEMCORR_BOOTSTRAP_PY=%%P"
    )
)

if not defined SEMCORR_BOOTSTRAP_PY (
    echo Error: Python 3.11 or newer was not found.
    echo Install Python from https://www.python.org/downloads/ and enable "Add Python to PATH".
    exit /b 1
)

"%SEMCORR_BOOTSTRAP_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo Error: SEM Map Corrector requires Python 3.11 or newer.
    "%SEMCORR_BOOTSTRAP_PY%" --version
    exit /b 1
)

if not exist "%SEMCORR_VENV_PY%" (
    echo First run: creating the SEM Corrector environment...
    "%SEMCORR_BOOTSTRAP_PY%" -m venv --system-site-packages "%SEMCORR_VENV%"
    if errorlevel 1 exit /b 1
)

"%SEMCORR_VENV_PY%" -c "import numpy, cv2, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    "%SEMCORR_VENV_PY%" -m pip install "numpy>=1.26,<3" "opencv-python-headless>=4.9,<6" "matplotlib>=3.8,<4"
    if errorlevel 1 exit /b 1
)

set "PYTHONPATH=%SEMCORR_ROOT%src;%PYTHONPATH%"
set "MPLCONFIGDIR=%SEMCORR_VENV%\matplotlib"
set "XDG_CACHE_HOME=%SEMCORR_VENV%\cache"

"%SEMCORR_VENV_PY%" -m semcorr %*
set "SEMCORR_EXIT=%ERRORLEVEL%"
endlocal & exit /b %SEMCORR_EXIT%
