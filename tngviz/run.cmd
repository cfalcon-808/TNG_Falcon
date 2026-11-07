@echo off
setlocal
rem Keep relative --tables and --game paths relative to the caller's directory.
set "TNGVIZ_PYTHON=%~dp0.venv-windows\Scripts\python.exe"
if not exist "%TNGVIZ_PYTHON%" (
    echo Windows environment missing. Run these commands in the tngviz folder:
    echo   py -3 -m venv .venv-windows
    echo   .venv-windows\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0..;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0.."
)
"%TNGVIZ_PYTHON%" -P -m tngviz %*
exit /b %errorlevel%
