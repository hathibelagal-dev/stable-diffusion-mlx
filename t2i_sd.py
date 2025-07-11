# Copyright © 2025 Ashraff Hathibelagal.

import mlx.core as mx
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
from mlx import nn
import os
import argparse

from stable_diffusion import StableDiffusion

def cli_mode():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="output file name", default="output.png")
    parser.add_argument("-n", "--no-novelai", help="don't use NovelAI anime model", action="store_true")
    args = parser.parse_args()

    if args.no_novelai:
        main_model_id = "CompVis/stable-diffusion-v1-4"
    else:
        main_model_id = "NovelAI/nai-anime-v2"

    print(f"Loading model {main_model_id}...")
    sd = StableDiffusion(
        unet_model_id=main_model_id,
        vae_model_id=main_model_id,
        float16=True,
    )
    nn.quantize(sd.unet, group_size=32, bits=8)
    sd.ensure_models_are_loaded()
    print("Loaded model successfully!")
    cfg = float(os.environ.get("SD_CFG", 7.5))
    print(f"Using cfg weight: {cfg}")
    steps = int(os.environ.get("SD_STEPS", 50))
    print(f"Using {steps} steps for denoising.")
    seed = int(os.environ.get("SD_SEED", 0))
    if seed == 0:
        seed = random.randint(0, 2**32 - 1)
    print(f"Using seed: {seed}")
    prompt = os.environ.get(
        "SD_PROMPT",
        "portrait of a cute kitten, leonardo da vinci style"
    )
    print(f"Using prompt: {prompt}")
    negative_prompt = os.environ.get(
        "SD_NEGATIVE_PROMPT",
        "missing finger, extra digits, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    )
    print(f"Using negative prompt: {negative_prompt}")

    latents = sd.generate_latents(
        prompt,
        n_images=1,
        cfg_weight=cfg,
        num_steps=steps,
        seed=seed,
        negative_text=negative_prompt,
    )
    print("Generating image...")
    for x_t in tqdm(latents, total=steps):
        mx.eval(x_t)

    del sd.text_encoder
    del sd.unet
    del sd.sampler
    peak_mem_unet = mx.get_peak_memory() / 1024**3

    decoded = []
    for i in tqdm(range(0, 1, 1)):
        decoded.append(sd.decode(x_t[i : i + 1]))
        mx.eval(decoded[-1])
    peak_mem_overall = mx.get_peak_memory() / 1024**3

    x = mx.concatenate(decoded, axis=0)
    x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
    B, H, W, C = x.shape
    x = x.reshape(1, B // 1, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(1 * H, B // 1 * W, C)
    x = (x * 255).astype(mx.uint8)

    print(f"Peak memory used for the unet: {peak_mem_unet:.3f}GB")
    print(f"Peak memory used overall:      {peak_mem_overall:.3f}GB")

    im = Image.fromarray(np.array(x))
    print(f"Saving image as {args.output}.")
    im.save(args.output)

if __name__ == "__main__":
    cli_mode()

