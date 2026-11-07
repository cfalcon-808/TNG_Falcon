@echo off
setlocal
rem Keep relative --tables and --game paths relative to the caller's directory.
set "TNGVIZ_PYTHON=%~dp0.venv-windows\Scripts\python.exe"
if not exist "%TNGVIZ_PYTHON%" (
    echo Windows environment missing. Create it with:
    echo   "%~dp0setup.cmd"              ^(or --no-models to skip the trained models^)
    exit /b 1
)
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0..;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0.."
)
"%TNGVIZ_PYTHON%" -P -m tngviz %*
exit /b %errorlevel%
