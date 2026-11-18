# Model & weight notices

EAIW code is released for research/experimental use. Model weights are **not**
in this repository; `scripts/download_all_models.py` downloads them from their
upstream publishers. Each download remains subject to its upstream license.

| Component | Upstream | License (upstream, verify before redistribution) |
|---|---|---|
| GoogLeNet / VGG-19 | PyTorch torchvision weights | BSD-3-Clause (torchvision) |
| Big-LaMa FFC | Carve/LaMa-ONNX, advimman/lama | Apache-2.0 |
| CycleGAN | junyanz/pytorch-CycleGAN-and-pix2pix | BSD-2-Clause |
| Real-ESRGAN | xinntao/Real-ESRGAN | BSD-3-Clause |
| VQGAN | CompVis/taming-transformers | MIT (code); check weight card |
| Stable Diffusion 1.4 | CompVis/stable-diffusion-v1-4 | CreativeML OpenRAIL-M |
| ModelScope text-to-video 1.7B | ali-vilab/text-to-video-ms-1.7b | Apache-2.0 (check model card) |
| Qwen3-VL-2B-Instruct | Qwen/Qwen3-VL-2B-Instruct | Apache-2.0 |
| CLIP ViT-L/14 | openai/clip-vit-large-patch14 | MIT |

Notes:

- SD1.3 is gated and not downloaded; lab 05 uses the public SD1.4 checkpoint as
  a same-architecture 2022 stand-in. Do not describe it as SD1.3.
- Qwen3.6-27B is not included or reproduced. Transformer Dream uses
  Qwen3-VL-2B with relative-depth layer mapping; results are not the 27B
  L60 outputs.
- If you redistribute a downloaded checkpoint, include the upstream LICENSE
  and model card.
