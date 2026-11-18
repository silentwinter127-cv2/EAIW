# Transformer Dream Engine (EAIW)

Qwen3-VL-2B 内部特征 → 层级 Lens 读出 → 输入像素激活最大化。

这是 `D:\Downloads\1.txt` 方案的 **可运行 P0 子系统**，独立于 EAIW 的 8 个历史实验室，避免把 2B 模型的梯度图塞进原有的 1.1MB 单页 UI。

- 页面：`http://127.0.0.1:7866`
- 启动：双击 `E:\EAIW\run_transformer_dream.bat`，或：
  ```bat
  cd /d E:\EAIW
  python -m transformer_dream.server --port 7866
  ```
- 模型：`E:\EAIW\models\qwen3-vl-2b-instruct`（Qwen/Qwen3-VL-2B-Instruct，Apache-2.0）
- 辅助：`E:\EAIW\models\clip-vit-large-patch14`（CLIP ViT-L/14，约 1.7GB，可选）
- 输出：`E:\EAIW\outputs\transformer_dream\run_*`（final.png、frame_*.jpg、meta.json）

## 为什么是 Qwen3-VL-2B

Qwen3.6-27B 的 64 层在 12GB 显存上无法做输入梯度反传。Qwen3-VL-2B 有：

- 28 层语言 Transformer / 2048 hidden；
- 24 层视觉编码器 / 1024 hidden；
- 4.26GB safetensors，fp16 实测占用 **4.3–5.0 GB 显存**。

`L_small = L_large / 64 × 28` 只表示相对深度，不代表相同语义位置。页面把
`M-Layer 01–28` 与 `V-Layer 01–24` 分开编号。

## 四个核心能力

### 1. Hidden State / Vision Feature

- 语言层：`output_hidden_states=True`，取 `hidden_states[L]` 的 image token 或 prompt token。
- 视觉层：forward hook 到 `model.visual.blocks[L-1]`，取 `[256,1024]` 特征。
- 不保存全量 attention 矩阵；Lens 只在 prompt token 上做投影；页面只拿 top-k。

### 2. Lens Readout

- 对 residual stream 应用最终 `language_model.norm`，再乘 `lm_head`，得到词表分布。
- 这是 **Logit Lens 近似**，不是模型内部真实自然语言标签，也不能证明存在“猫神经元”。
- 页面显示 28 层 top-3；默认读 prompt 最后一个 token（如 `crown`），不是 assistant token。

### 3. Activation Maximization

图像 `[1,3,S,S]` 是唯一可训练张量；Qwen 权重全部 `requires_grad_(False)`。
每个 step：

1. 可微 patchify 成 Qwen2VL 的 `[N, 1536]` patch 序列；
2. 冻结模型 forward；
3. 取目标层 image token / visual block 特征；
4. 反传回像素，Adam 更新；
5. TV 正则 + 小幅 jitter，避免锁死在像素网格。

三种目标：

| 模式 | 目标 | 视觉结果 |
|---|---|---|
| `direction` | 目标文本与 baseline 文本在 M 层的隐藏状态差方向 | 语义纹理、文字状伪影 |
| `logit` | 目标词在指定层的 Lens logit | 更贴近目标词的碎片 |
| `neuron` | M 层单通道均值 | 局部纹理/颜色幻觉 |
| `vision` | V 层单通道均值 | 更有机、少文字、偏神经纹理 |
| `clip` | CLIP 图文余弦相似度（辅助基线） | 更像目标概念、颜色纹理强 |

### 4. 实验档案

每次运行保存 `final.png`、10 张过程帧、`meta.json`（prompt / layer / seed / loss / lens）。页面显示本次输出目录。

## 低资源约束

- **C: 只读**：所有 `HF_HOME`、`TORCH_HOME`、`TEMP`、`TMP`、输出都在 E:。
- 已在 E: 增加 16GB `pagefile.sys`；C: 页面文件仍存在但不再增长。没有这个页面文件，4.3GB safetensors 的 `safe_open` 会报 `os error 1455`。
- 单任务串行，Qwen 常驻约 4.3GB；`Release VRAM` 可卸载。
- 默认 256×256；320/448 已实测可用（448 约 5.0GB）。448 以上不开放，避免 12GB 机器 OOM。
- 默认 **M-Layer 24 / 240 steps / LR 0.02 / TV 2.0**；448 约 50–70 秒（Qwen），CLIP 200 步约 7–10 秒。
- **CLIP ViT-L/14 可选模式**：选择 `CLIP ViT-L/14` 时会先卸载 Qwen，再加载 1.7GB CLIP；不会同时占两张模型。CLIP 模式生成语义纹理，不输出 Qwen 的 Lens 表。
- 首次使用 CLIP 前需要下载：`python -m transformer_dream.download_clip`（约 1.7GB，全部写入 E:）。

## HTTP API

```text
GET  /api/status
POST /api/run       {prompt, mode, layer, vision_layer, channel,
                     target_word, negative, steps, size, lr,
                     tv_weight, seed, init_image_b64}
GET  /api/job/<id>
POST /api/extract   {prompt, size, vision_layers}
POST /api/unload
```

`/api/run` 返回 job id；前端每 700ms 轮询 `/api/job`，完成后拿到 final、frames、lens_table。

## 已验证

- 页面加载、Start Dream、Lens、进度、帧序列、28 层 Lens 表格；
- Qwen 4.3GB fp16，256/320/448 三档均成功；
- 典型 240 steps / 256px 约 15–20 秒；
- 服务器：`http://127.0.0.1:7866`。

## 文件结构

```text
transformer_dream/
├── bootstrap.py        # 在 import torch 前把缓存/临时目录锁到 E:
├── config.py
├── image_ops.py        # 可微 patchify / decode / TV / jitter
├── lens.py             # Logit Lens
├── model_runtime.py    # Qwen 加载、forward、vision hook
├── engine.py           # 激活最大化 + 实验档案
├── server.py           # stdlib HTTP + job 队列
├── download_model.py   # 只下载到 E:
└── web/index.html      # 独立实验台
```

## 为什么第一版图不够“丰富”

旧版直接对像素做 Adam，只有轻微 jitter + 弱 TV，且默认 M-Layer 16。这样优化器会利用 Qwen 的高频对抗方向，结果像密集文字/噪点，而不是截图里的多尺度结构。现在改为：

- **默认 L24**（更接近 L60 的相对深度），而不是中层的 L16；
- 每步随机 **缩放/旋转/平移/水平翻转**，让同一个概念在不同尺度重复出现；
- 周期性 3×3 blur，抑制像素网格噪声；
- 随机颜色去相关，避免单色塌缩；
- **梯度按平均绝对值归一化**，让不同层的梯度尺度一致；
- TV 提高到 2.0；
- 240 步，而不是 120 步；
- 需要“猫/王冠/蘑菇”这类可辨认对象时，选 **CLIP ViT-L/14** 模式，它比 2B 的 M-Layer 激活更接近截图里的语义图。

## 边界

- 这不是 Qwen3.6-27B 的逐像素复刻，层号不能直接等同；
- 2B 模型的视觉语义弱于 27B，`crown` 等概念更多表现为纹理/文字碎片；
- 若需要截图里更“像猫”的结果，优先用 CLIP 模式；它是 CLIP 语义先验，不是 Qwen 神经元激活，不能混为一谈。
