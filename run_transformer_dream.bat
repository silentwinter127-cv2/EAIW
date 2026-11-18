@echo off
setlocal
cd /d "%~dp0"
set HF_HOME=%~dp0models\hf_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TRANSFORMERS_OFFLINE=1
set HF_HUB_OFFLINE=1
netstat -ano | findstr /R /C:":7866 .*LISTENING" >nul
if %errorlevel%==0 (
  start "" http://127.0.0.1:7866
  exit /b 0
)
start "" http://127.0.0.1:7866
python -m transformer_dream.server --port 7866
