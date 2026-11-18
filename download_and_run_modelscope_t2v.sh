#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "============================================================================"
echo "EAIW 1.0 - 下载 2023 ModelScope 1.7B 文生视频模型到 models/weights/"
echo "仓库: ali-vilab/text-to-video-ms-1.7b（ModelScope / hf-mirror / HuggingFace 自动回退）"
echo "共 12 个 fp16 文件，约 3.42 GB；支持断点续传与 SHA-256 校验"
echo "============================================================================"
if ! python3 scripts/download_historical_weights.py --model modelscope_t2v_2023 --jobs 3; then
    echo
    echo "[错误] 模型下载失败。请检查网络后重新运行本脚本，已下载的分片会自动续传。"
    exit 1
fi

echo
echo "============================================================================"
echo "下载完成，正在启动 EAIW 1.0 本地服务 http://127.0.0.1:7865 ..."
echo "============================================================================"
(sleep 3 && python3 -m webbrowser "http://127.0.0.1:7865" >/dev/null 2>&1) &
python3 main.py --server
