@echo off
rem Double-click to install everything doc-to-audiobook needs.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if "%~1"=="" pause
