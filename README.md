# Stable Diffusion on MLX (SD 1.4 / SD 1.5)

Run Stable Diffusion on Apple Silicon using MLX. This repo focuses on SD 1.x models, with a fast, minimal pipeline and a simple CLI. By default it loads **NovelAI/nai-anime-v2** UNet + VAE with the SD 1.4 text stack, but you can also run most SD1.5 checkpoints from CivitAI with a single file.

**Highlights**
- SD 1.4 + SD 1.5 model support on MLX
- Works great on older Macs
- CivitAI `.safetensors` SD1.5 checkpoint loading (UNet + text encoder)
- Optional custom VAE from checkpoint
- Simple environment-variable driven prompts

---

## Quickstart

Install deps and generate your first image:

```bash
pip install -r requirements.txt

SD_PROMPT="flowers, flower field, sunset, no humans" \
SD_NEGATIVE_PROMPT="humans" \
SD_SEED=56 \
python3 t2i_sd.py
```

Output defaults to `output.png` (use `-o` to change the filename).

---

## Use a CivitAI SD1.5 Checkpoint

Most SD1.5 models from CivitAI are now supported. Download a SD1.5 `.safetensors` file and run:

```bash
python3 t2i_sd.py --ckpt /path/to/model.safetensors
```

If you need a different base config/tokenizer:

```bash
python3 t2i_sd.py --ckpt /path/to/model.safetensors --base-model runwayml/stable-diffusion-v1-5
```

By default the VAE comes from the base model for max compatibility. To use the checkpoint’s VAE:

```bash
python3 t2i_sd.py --ckpt /path/to/model.safetensors --use-ckpt-vae
```

---

## CLI Options

- `-o output.png`: output filename
- `-n, --no-novelai`: use `CompVis/stable-diffusion-v1-4`
- `--ckpt /path/to/model.safetensors`: load SD1.5 checkpoint
- `--base-model runwayml/stable-diffusion-v1-5`: base configs/tokenizer
- `--use-ckpt-vae`: use the checkpoint’s VAE

---

## Prompt Controls (Environment Variables)

- `SD_PROMPT`: positive prompt
- `SD_NEGATIVE_PROMPT`: negative prompt
- `SD_CFG`: classifier-free guidance (default 7.5)
- `SD_STEPS`: denoising steps (default 50)
- `SD_SEED`: seed (0 picks a random one)

Example:

```bash
SD_PROMPT="1girl, cinematic lighting" \
SD_NEGATIVE_PROMPT="ugly, disfigured" \
SD_CFG=7.0 SD_STEPS=28 SD_SEED=123 \
python3 t2i_sd.py --ckpt /path/to/model.safetensors
```

---

## Example Outputs

With `SD_STEPS` around 28 you can get strong results quickly; 50 steps is the default for higher detail.

<img src="https://github.com/user-attachments/assets/d0d851f2-8319-41d6-894c-73ce50317028" style="width:256px"/>
<img src="https://github.com/user-attachments/assets/95ec53f5-1823-4bf6-b988-78c4d950fb5e" style="width:256px"/>
<img src="https://github.com/user-attachments/assets/8c539f64-b536-4fbe-b598-df6bb491cb6a" style="width:256px"/>

---

## Notes

- SD 1.4 is especially strong for stylized artwork.
- SD 1.5 shares the same CLIP ViT-L/14 text encoder, which this repo uses.
- SD2/SDXL checkpoints are not supported.
