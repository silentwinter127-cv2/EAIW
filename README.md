# EAIW 1.0 — Early AI Visual Workstation (早期 AI 视觉工作站)

一套复现、研究与模拟 **2014–2022 年早期 AI 视觉技术演进及其典型视觉故障（AI Artifacts）** 的模块化工作站系统。可作为既有数字媒介仿真系统（摄像机、磁带、DV、无线电、字幕机）的独立视觉子系统运行。

## 一、核心架构与目录结构

```text
EAIW/
├── main.py                      # 主程序入口（支持 --gui 桌面界面、--demo 核心工作流演练、--inspect-db 档案检索）
├── requirements.txt             # Python 依赖清单
├── pyproject.toml               # 标准 Python 打包配置
├── core/
│   ├── engine.py                # 统一调度引擎（支持 Historical / Modern Reproduction / Simulation 三模式）
│   ├── model_manager.py         # 模型注册表、SHA256 校验、许可证记录与子进程隔离调用器
│   ├── pipeline.py              # 节点式工作流编排器（Input -> Preprocess -> Lab Node -> Post-process -> Archive）
│   ├── image_io.py              # 统一图像读写、遮罩膨胀/羽化/精度退化处理与 PNG 元数据嵌入
│   ├── job_scheduler.py         # P0/P1 优先级批处理任务队列
│   └── metadata.py              # SQLite 实验记录系统、八类 AI 视觉故障数据库与 PSNR/SSIM/LPIPS/CLIP/MBE 评价指标
├── laboratories/
│   ├── neural_vision/           # 01 神经网络视觉实验室（DeepDream 多尺度层级激活、Gatys NST）
│   ├── gan/                     # 02 GAN 图像生成实验室（DCGAN、StyleGAN 潜空间漫游、Slerp 插值、Style Mixing、Truncation）
│   ├── inpainting/              # 03 AI 图像修复实验室（Telea/NS/PatchMatch、DeepFill v2、LaMa、SD Inpainting + 7 维故障模拟器）
│   ├── image_translation/       # 04 图像转换实验室（Pix2Pix / CycleGAN：夏->冬、日->夜、马->斑马、照片->卡通、线稿->建筑）
│   ├── text_to_image/           # 05 文本生成实验室（VQGAN+CLIP 原生控制台、Disco Diffusion v5、SD 1.x）
│   └── postprocessing/          # 06 视觉后处理实验室（ESRGAN、GFPGAN 人脸修复、DeOldify 着色及伪细节差异热力图）
├── models/
│   ├── registry.json            # 2014–2022 历史模型版本、SHA256、来源与原生参数规范
│   ├── sd_downloader.py         # SD1.4 Diffusers 分片下载 + SHA 校验
│   └── worker_runner.py         # 遵循统一 JSON 推理协议的独立子进程推理执行器
├── scripts/
│   ├── download_all_models.py   # 一键补全所有运行时权重（推荐）
│   └── download_historical_weights.py  # 历史权重 + ModelScope T2V
├── transformer_dream/           # Qwen3-VL-2B 层级 Lens + 激活最大化实验台（7866）
├── workflows/                   # 1.0 核心验收工作流（A: 2015 DeepDream, B: 四代修复对照, C: VQGAN+CLIP）
├── presets/                     # 手机 AI 消除故障、PatchMatch 纹理重复、CNN 接缝断层预设
├── datasets/testing/            # 内置测试场景与遮罩样本（会提交）
├── ui/                          # Web 工作站单页（ui/web_workstation.html）
├── tests/                       # 单元测试与端到端验收测试套件
└── archive/                     # 本地实验产物；已在 .gitignore 中排除
```

`models/weights/`、Qwen/CLIP 权重、`archive/`、`outputs/`、`temp/` 均不提交；由下载程序补全。

## 二、模型下载（新克隆必读）

```bash
# 1) 安装依赖（torch 需按本机 CUDA 版本先装）
python -m pip install -r requirements.txt

# 2) 一键下载全部运行时模型到 EAIW/models/
python scripts/download_all_models.py
```

下载内容：

| 模型 | 用途 | 约大小 |
|---|---|---:|
| GoogLeNet / VGG-19 / CycleGAN ×2 / LaMa / Real-ESRGAN / VQGAN | 01/03/04/06 真实历史算子 | 1.8 GB |
| ModelScope `text-to-video-ms-1.7b` | 08 文生视频 | 3.42 GB |
| Stable Diffusion 1.4 Diffusers | 05 文生图（SD1.3 受限替代） | 3.97 GB |
| Qwen3-VL-2B-Instruct | Transformer Dream | 4.26 GB |
| CLIP ViT-L/14 | Transformer Dream 辅助语义 | 1.71 GB |

`--all` 会跳过需要 HF 登录的 SD1.3；05 实际使用公开的 SD1.4。所有文件均写入 `E:\EAIW\models\`，不写 C:。

## 三、快速启动

### 1. 运行三大核心验收工作流与数据库构建
```bash
python3 main.py --demo --inspect-db
```

### 2. 启动 PySide6 专业桌面工作站界面
```bash
pip install -r requirements.txt
python3 main.py --gui
```

### 3. 运行自动化测试套件
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 4. 通过统一 JSON 协议调用独立子进程推理
```bash
python3 models/worker_runner.py << 'EOF'
{
  "model_id": "deepfill_v2_2019",
  "execution_mode": "Simulation",
  "operation": "inpaint",
  "input": "datasets/testing/sample_street_scene.png",
  "mask": "datasets/testing/sample_street_mask.png",
  "parameters": {
    "seed": 42,
    "semantic_consistency": 0.35,
    "spatial_consistency": 0.40,
    "boundary_blending": 0.50
  },
  "output": "archive/protocol_inpaint_result.png"
}
EOF
```

## 四、接入真实历史模型权重与本地推理服务 (Real Historical Models)

本项目在 `core/real_models_backend.py` 与 `scripts/download_historical_weights.py` 中内置了 **2014–2023 年真实历史模型的原生算子与权重加载器**：

1. **下载 2023 ModelScope 文生视频模型（推荐，12 个 fp16 文件，约 3.42 GB）**：
```bash
# 命令行下载（Windows / macOS / Linux 通用）
python3 scripts/download_historical_weights.py --model modelscope_t2v_2023 --jobs 3
# 或直接双击 Windows 一键脚本（下载完成后自动启动服务）
download_and_run_modelscope_t2v.bat
```
- 官方仓库为 `ali-vilab/text-to-video-ms-1.7b`（旧的 `damo-vilab/text-to-video-ms-1.7b` 会自动 301 到新地址）。
- 下载器会依次尝试 `modelscope.cn` → `hf-mirror.com` → `huggingface.co`，某个镜像失败自动切换（ModelScope CDN 在国内通常最快）。
- 默认保存到 `models/weights/text-to-video-ms-1.7b/`；可用环境变量 `EAIW_T2V_MODEL_DIR` 指向其它目录。
- 支持断点续传：文件先写入 `.part`，中断后重新执行同一命令即可从断点继续；完成后按**精确字节数 + SHA-256** 校验，校验失败会删除坏文件并重试。
- 大文件并行下载（`--jobs`，默认 3，最大 6），可显著缩短 3.42 GB 权重的下载时间。
- 下载内容：`model_index.json`、`scheduler/scheduler_config.json`、`text_encoder/config.json` + `model.fp16.safetensors`、`tokenizer/{merges.txt,vocab.json,tokenizer_config.json,special_tokens_map.json}`、`unet/config.json` + `diffusion_pytorch_model.fp16.safetensors`、`vae/config.json` + `diffusion_pytorch_model.fp16.safetensors`。

2. **校验与查看状态**：
```bash
python3 scripts/download_historical_weights.py --list          # 列出全部历史权重及本地状态
python3 scripts/download_historical_weights.py --verify        # 深度 SHA-256 校验文生视频模型
python3 scripts/download_historical_weights.py --verify KEY    # 校验指定单文件权重
```
`--list` 不会触发下载；`--verify` 会读取全部 12 个文件并逐字节校验 SHA-256。

3. **下载其余历史模型权重（可选，文生视频不需要）**：
```bash
python3 scripts/download_historical_weights.py --all
```
包括：
- `googlenet-1378be20.pth`（2015 BVLC GoogLeNet / Inception v1 真实 Forward Hook 多尺度 Octave 梯度上升 DeepDream）
- `vgg19-dcbb9e9d.pth`（2015 Gatys et al. VGG-19 Gram 矩阵 L-BFGS 神经风格迁移）
- `builtin://patchmatch-telea-ns`（2004 Telea FMM、2001 Navier-Stokes PDE、以及完整的 2009 Barnes et al. PatchMatch 双金字塔随机近邻场块搜索修复）
- `lama_fp32.onnx`（2021 Suvorov et al. Big-LaMa 快速傅里叶卷积大感受野图像修复）
- `summer2winter_yosemite.pth` / `horse2zebra.pth`（2017 UC Berkeley 官方 CycleGAN / Pix2Pix 权重）
- `vqgan_imagenet_f16_16384.ckpt` + OpenAI `CLIP ViT-B/32`（2021 Esser & Crowson 32-Cutouts 离散码本梯度优化器）
- `kuprel/min-dalle` & `CompVis/stable-diffusion-v1-4` / `v1-1`（2022 早期自回归 dVAE 与 Latent Diffusion 权重，含每步 VAE 潜空间解码回调）
- `RealESRGAN_x4plus.pth`（2021 RRDBNet 超分辨率与伪细节审计）

4. **文生视频推理说明（已配置 RTX 3060 GPU + fp16）**：
- 下载完成后，`core/real_models_backend.py` 会直接离线加载上面的本地目录（`variant="fp16"`）并缓存管线，不再访问网络。
- 当前环境已安装 `torch 2.14.0+cu130`，检测到 `NVIDIA GeForce RTX 3060 12GB`；有 CUDA 时自动使用 **fp16 + 整卡加载**，16 帧实测约 **19 秒**（含首次加载约 5 秒），显存峰值约 4.1GB。
- 重装或升级 GPU 版 PyTorch 时执行（不要装默认的 `+cpu` 轮子）：
```powershell
python -m pip install torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
- 显存较小的显卡可设 `EAIW_T2V_OFFLOAD=1`，退回逐层 CPU offload（更省显存、但明显更慢）；显存紧张时还可设 `EAIW_T2V_ATTN_SLICING=1` 开启注意力切片。
- **纯 CPU 环境默认跳过真实推理**（16 帧 3D-UNet 约需 7–10 分钟，会长时间占用网页请求，也会拖慢单元测试），此时工作站自动使用内置参考帧。
- 确需在 CPU 上跑真实模型时设置 `EAIW_T2V_CPU=1`：
```bash
set EAIW_T2V_CPU=1        # Windows cmd
$env:EAIW_T2V_CPU="1"     # PowerShell
export EAIW_T2V_CPU=1     # bash
```
- 另需安装推理依赖：`pip install diffusers transformers accelerate safetensors`（下载脚本可用 `--install-deps` 自动安装）。

5. **启动本地真实模型 HTTP 桥接服务 (`http://127.0.0.1:7865`) 与网页端实时联动**：
```bash
python3 main.py --server
```
启动后，在网页工作站点击顶部 **「🔌 真实历史模型引擎 (TF.js / PyTorch / HF)」 -> 「探测并连接本地服务」**。连接成功后，桥接卡片会显示本地 1.7B 模型状态：

- `GET /api/status` 的 `capabilities.t2v_model` 会返回 `available`（12 个权重是否齐全）、`deps_ready`（diffusers/transformers/accelerate 是否安装）、`device`、`enabled` 与 `reason`。
- 前端生成文生视频时会调用 `POST /api/live_prompt_frames`，并带上 `use_real_model: true`；后端优先执行本地 `ali-vilab/text-to-video-ms-1.7b` Diffusers fp16 管线，返回 16 帧真实视频帧（`real_model: true`、`source: 本地 ali-vilab/text-to-video-ms-1.7b`）。
- 生成由 **「⚡ 生成动画」按钮 / Enter** 触发（真实 GPU 请求约 20 秒）；输入提示词时不会自动排队请求，避免一次输入触发两次 GPU 任务。生成期间画布显示明确的“正在生成”占位画面，而不是内置 Panda / 实景检索演示帧；只有收到并校验 16 帧 `real_model: true` 后才显示“真实视频生成”完成状态。
- 权重或依赖不完整、或 GPU 不可用时，接口会返回 `real_model: false` 并自动回退到实景检索合成，不会让页面白屏；此时状态栏会明确标注 **“已回退到演示序列（非真实模型）”**，不会把演示帧标成真实模型输出。
- **GPU 机器**：连接本地服务后 `t2v_model.enabled` 自动为 `true`，前端复选框默认勾选并直接走 fp16 GPU 推理，无需设置任何环境变量。
- **纯 CPU 机器**：本地模型默认不自动运行（16 帧可能需要约 10 分钟）。连接后勾选桥接卡片里的 **“文生视频使用本地 1.7B 模型（真实 Diffusers 推理）”**，即可强制使用 CPU 真实推理；首次加载约 8 秒，之后复用缓存管线。取消勾选则使用快速合成预览。
- **DeepDream（2015 BVLC GoogLeNet）**：在左侧选择“神经网络视觉实验室”与 `deepdream_googlenet_2015`，点击“执行当前工作站节点”或“⚡ 运行真实算子”。前端会调用 `POST /api/infer`（`layer` / `octaves` 参数生效），后端优先加载项目内 `models/weights/googlenet-1378be20.pth`，执行 `inception_4c` 真实特征梯度上升，返回 `output_base64` 后直接替换画布与迭代序列；不再使用浏览器 TF.js 随机卷积近似，也不会只显示 OpenCV 漩涡。权重缺失时才回退到演示算法，并在状态栏明确提示。
- **输入图 → 密集“狗脸/动物脸”效果**：输入图模式现在走经典 octave 金字塔——最长边从 128 开始，按约 1.4 倍逐级放大到原图（4K 自动限制到 1024），每级叠加原图高频细节；梯度按平均绝对值归一化，并加入随机抖动，再以 10–24 次迭代累积特征。这样原图不会被简单磨皮，而会在毛发、草丛、云层、砖墙等纹理处成片“长出”重复的动物脸和眼睛；这也是你给的两张经典 DeepDream 图的形成方式。图越大细节越多，512 约 1–2 秒，1024 约 3–5 秒。
- **导出高清图片**：画布下方工具栏新增 **“⬇ 导出高清图片”**。上传图片后执行 DeepDream，按钮会导出后端返回的原始分辨率 PNG（例如 1024×1024），而不是画布上被缩小的 320×320 预览；没有原始输出时退回导出当前结果帧，并在提示里标明实际尺寸。文件名包含模型、时间和分辨率，例如 `eaiw_deepdream_googlenet_2015_2026-11-09T13-43-18_1024x1024.png`。上传图的最长边会以原图送入后端并限制在 1024，避免显存不足。
- **DeepDream 从零生成（无输入图）**：勾选参数台里的 **“从零生成（随机噪声 → 多尺度 DeepDream，无需输入图）”**，可选 512 / 768 / 1024 输出。后端不再需要任何原图：从 64×64 随机噪声开始，按 64→128→256→512（最大到 1024）逐级放大，低分辨率做更多次 `inception_4c` 特征梯度上升，高分辨率补细节，因此会先长出大尺度“幻觉生物/脸”，再叠加纹理。每次点击都会用当前 Seed 生成不同结果；迭代次数沿用“Steps”，GPU 上 512 约 1–3 秒、1024 约 2–5 秒。注意：从零 DeepDream 是随机优化，输出风格和具体生物由 Seed / 层 / 迭代决定，不保证与某张参考图逐像素相似；想要更接近参考图构图时，仍建议用“从输入图开始”的经典模式。
- **图生视频（Image-to-Video）**：在文生视频实验室上传/载入一张图片，桥接卡片会自动勾选 **“以当前输入图片作为视频首帧”**，可调“首帧保持强度”（0.15–0.65，默认 0.35）。前端把图片以 `image_base64` + `init_strength` 发送到 `/api/live_prompt_frames`；后端先用 VAE 编码该图、按 strength 加噪并初始化首帧 latent，再跑本地 1.7B 管线，最后把第 1 帧替换为原图并对前 4 帧做交叉淡化。RTX 3060 上 16 帧约 10–20 秒。注意：这是基于 `TextToVideoSDPipeline` 的首帧条件扩展，而非单独训练的 I2V 模型；强度越低越贴近原图，越高运动幅度越大但可能出现噪点。
- **03 AI 图像修复（真实 LaMa + 手机 AI 消除复刻）**：选择“03 AI 图像修复 / 消除”，左侧载入或上传图片，在红色遮罩层涂抹要消除的区域，模型下拉选择 `lama_ffc_2021`，点“执行修复”。默认勾选 **“手机 AI 消除（生成式重绘，可能变脸）”**，会像 OPPO 手机消除那样重绘整张合影：人物被替换为新的卡通角色，邻近熊/猴的脸也会被重绘成“哈人”效果；取消勾选则切回干净的 LaMa 真实补全。后端会加载本地 `models/weights/lama_fp32.onnx`（Big-LaMa FFC，512×512），真实补全遮罩区域，不再用 `effect_4_dalle_clay.jpg` 的假黏土补丁盖住结果；首次调用约 12 秒（ONNX session 初始化），之后每次约 1 秒；默认遮罩已收窄到人物主体，取消遮罩叠加后即可查看“AI 消除”补全结果。
- **05 文生图（SD1.4 本地扩散）**：官方 `CompVis/stable-diffusion-v1-3` 已下架/受限，项目下载公开的 SD1.4 作为同代替代（界面仍保留 SD1.3 历史模型名）。先下载 3.97GB 权重：
```bash
python -m models.sd_downloader
```
默认保存到 `models/weights/stable-diffusion-v1-4/`。进入“05 文本生成实验室”后，顶部会出现 **“🖼 生成图片”** 按钮和 **正向提示词 / 反向提示词** 两栏；正向写主体与风格，反向写 `lowres, blurry, watermark, text, worst quality` 等排除项。点按钮后本地 fp16 管线生成图片，默认 **512×512**（25 steps、CFG 7.5，RTX 3060 约 3–8 秒），输出就是 2022 年 SD1.x 的画质；右侧参数台新增 **“画幅尺寸 Canvas Size”**，可切换 512 / 768 / 1024（SD1.x 在 768 以上可能出现结构崩坏，属模型年代限制），最大画幅约 3–5 秒。可再用“导出高清图片”保存当前画幅的 PNG。

6. **下载 HTTP API**（便于前端或脚本触发）：
```bash
# 查询状态：文生视频 12 文件进度 + 全部单文件权重状态 + 后台任务
curl http://127.0.0.1:7865/api/download_status

# 触发下载（立即返回，后台线程执行；t2v 可用 jobs 控制并行数）
curl -X POST http://127.0.0.1:7865/api/download_weight \
     -H "Content-Type: application/json" \
     -d '{"model_key":"modelscope_t2v_2023","jobs":3}'

# 单文件权重同样走该接口，例如 {"model_key":"esrgan_realesrgan_x4plus_2021"}
```


## Transformer Dream Engine（Qwen3-VL-2B 激活可视化）

独立的 Qwen3-VL 内部特征实验台，复现“读取 Transformer 层 → Lens 读出 → 输入像素激活最大化生成 Dream 图像”的流程：

- 页面：`http://127.0.0.1:7866`
- 启动：`run_transformer_dream.bat` 或 `python -m transformer_dream.server --port 7866`
- 模型：`models/qwen3-vl-2b-instruct`（4.26GB，fp16 约 4.3GB 显存）
- 支持：M-Layer 01–28 / V-Layer 01–24、Logit Lens、Feature Direction、单通道激活、可选 CLIP ViT-L/14 语义模式、256/320/448 画幅、过程帧与实验档案
- 缓存/临时/输出全部在 E:；C: 不写模型。详见 `transformer_dream/README.md`
