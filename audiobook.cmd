@echo off
rem Runs the converter with the core env that install.py created.
set "PY=%~dp0envs\core\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0envs\core\python.exe"
"%PY%" "%~dp0pdf_to_audiobook.py" %*
