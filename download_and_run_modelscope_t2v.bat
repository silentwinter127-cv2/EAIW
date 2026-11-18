@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================================
echo EAIW 1.0 - 下载 2023 ModelScope 1.7B 文生视频模型到 models\weights\
echo 仓库: ali-vilab/text-to-video-ms-1.7b（ModelScope / hf-mirror / HuggingFace 自动回退）
echo 共 12 个 fp16 文件，约 3.42 GB；支持断点续传与 SHA-256 校验
echo ============================================================================
python scripts\download_historical_weights.py --model modelscope_t2v_2023 --jobs 3
if errorlevel 1 (
    echo.
    echo [错误] 模型下载失败。请检查网络后重新运行本脚本，已下载的分片会自动续传。
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo 下载完成，正在启动 EAIW 1.0 本地服务 http://127.0.0.1:7865 ...
echo ============================================================================
start "EAIW Server" cmd /k "python main.py --server"
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:7865
echo 服务已在新窗口运行；关闭该窗口即可停止服务。
pause
