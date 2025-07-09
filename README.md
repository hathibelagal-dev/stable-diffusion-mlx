Stable Diffusion 1.4 in MLX
================

Run Stable Diffusion 1.4 using MLX.

By default, this repo uses the unet and vae of `NovelAI/nai-anime-v2`. The rest of the components come from `CompVis/stable-diffusion-v1-4`.

## Sample Usage

```bash
SD_PROMPT="photograph of windy landscape, rainy day, masterpiece" \
SD_NEGATIVE_PROMPT="humans" \
python3 t2i_sd.py
```

The above command will generate an image named `output.png`. You can use the `-o` parameter to change the output filename.