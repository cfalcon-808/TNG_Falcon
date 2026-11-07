@echo off
rem Create tngviz\.venv-windows and install the visualizer (Windows).
rem   tngviz\setup.cmd               board, perfect play and the trained models (CPU torch, about 200 MB)
rem   tngviz\setup.cmd --no-models   board and perfect play only
setlocal
set "MODELS=1"
if /i "%~1"=="--no-models" set "MODELS=0"
if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help
if not "%~1"=="" if /i not "%~1"=="--no-models" (
    echo unknown option: %~1 ^(use --no-models or --help^)
    exit /b 2
)
set "VENV=%~dp0.venv-windows"
if not defined PYTHON set "PYTHON=py -3"
%PYTHON% -m venv "%VENV%" || exit /b 1
set "PY=%VENV%\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip || exit /b 1
"%PY%" -m pip install -r "%~dp0requirements.txt" || exit /b 1
if "%MODELS%"=="1" (
    "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || exit /b 1
    "%PY%" -m pip install -r "%~dp0requirements-models.txt" || exit /b 1
)
echo Done. Start it with: %~dp0run.cmd
exit /b 0
:help
echo   tngviz\setup.cmd               board, perfect play and the trained models (CPU torch, about 200 MB)
echo   tngviz\setup.cmd --no-models   board and perfect play only
exit /b 0
