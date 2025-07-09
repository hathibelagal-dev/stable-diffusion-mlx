Stable Diffusion 1.4 in MLX
================

NovelAI has released the weights for their older SD1.5-based NovelAI Diffusion V2 anime model! I created this repo to be able to run **nai-anime-v2** on an old Mac using MLX.

So, by default, this repo uses the Unet and VAE of `NovelAI/nai-anime-v2`. The rest of the components come from `CompVis/stable-diffusion-v1-4`.

**Note:** SD 1.4 is particularly strong for generating stylized images and art. It uses the same text encoder as SD 1.5, CLIP ViT-L/14, and was trained on subsets of the LAION-5B dataset, specifically "laion-aesthetics v2 5+" for aesthetic quality.

## Sample Usage

Once you clone this repository and install the requirements, you can run the following command to generate an image:

```bash
SD_PROMPT="flowers, flower field, sunset, no humans" \
SD_NEGATIVE_PROMPT="humans" \
SD_SEED=56 \
python3 t2i_sd.py
```

The above command will generate an image named `output.png`. You can use the `-o` parameter to change the output filename.

## Example Outputs

The performance is decent. You can get a good image with `SD_STEPS` set to around 28. Even with `SD_STEPS` set to 50 (default), it only takes about 50 seconds to generate an image.

<img src="https://github.com/user-attachments/assets/d0d851f2-8319-41d6-894c-73ce50317028" style="width:256px"/>
<img src="https://github.com/user-attachments/assets/95ec53f5-1823-4bf6-b988-78c4d950fb5e" style="width:256px"/>
<img src="https://github.com/user-attachments/assets/8c539f64-b536-4fbe-b598-df6bb491cb6a" style="width:256px"/>
