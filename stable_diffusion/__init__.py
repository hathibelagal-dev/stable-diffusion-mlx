# Copyright © 2023-2024 Apple Inc.

import time
from typing import Optional, Tuple

import mlx.core as mx

from .model_io import (
    load_autoencoder,
    load_diffusion_config,
    load_text_encoder,
    load_tokenizer,
    load_unet,
)
from .sampler import SimpleEulerSampler


class StableDiffusion:
    def __init__(
        self,
        unet_model_id: str = "CompVis/stable-diffusion-v1-4",
        vae_model_id: str = "CompVis/stable-diffusion-v1-4",
        text_encoder_model_id: str = "CompVis/stable-diffusion-v1-4",
        tokenizer_model_id: str = "CompVis/stable-diffusion-v1-4",
        float16: bool = False,
    ):
        self.dtype = mx.float16 if float16 else mx.float32
        self.diffusion_config = load_diffusion_config(text_encoder_model_id)
        self.unet = load_unet(unet_model_id, float16)
        self.text_encoder = load_text_encoder(text_encoder_model_id, float16)
        self.autoencoder = load_autoencoder(vae_model_id, False)
        self.sampler = SimpleEulerSampler(self.diffusion_config)
        self.tokenizer = load_tokenizer(tokenizer_model_id)

    def ensure_models_are_loaded(self):
        mx.eval(self.unet.parameters())
        mx.eval(self.text_encoder.parameters())
        mx.eval(self.autoencoder.parameters())

    def _tokenize(self, tokenizer, text: str, negative_text: Optional[str] = None):
        # Tokenize the text
        tokens = [tokenizer.tokenize(text)]
        if negative_text is not None:
            tokens += [tokenizer.tokenize(negative_text)]
        lengths = [len(t) for t in tokens]
        N = max(lengths)
        tokens = [t + [0] * (N - len(t)) for t in tokens]
        tokens = mx.array(tokens)

        return tokens

    def _get_text_conditioning(
        self,
        text: str,
        n_images: int = 1,
        cfg_weight: float = 7.5,
        negative_text: str = "",
    ):
        # Tokenize the text
        tokens = self._tokenize(
            self.tokenizer, text, (negative_text if cfg_weight > 1 else None)
        )

        # Compute the features
        conditioning = self.text_encoder(tokens).last_hidden_state

        # Repeat the conditioning for each of the generated images
        if n_images > 1:
            conditioning = mx.repeat(conditioning, n_images, axis=0)

        return conditioning

    def _denoising_step(
        self, x_t, t, t_prev, conditioning, cfg_weight: float = 7.5, text_time=None
    ):
        x_t_unet = mx.concatenate([x_t] * 2, axis=0) if cfg_weight > 1 else x_t
        t_unet = mx.broadcast_to(t, [len(x_t_unet)])
        eps_pred = self.unet(
            x_t_unet, t_unet, encoder_x=conditioning, text_time=text_time
        )

        if cfg_weight > 1:
            eps_text, eps_neg = eps_pred.split(2)
            eps_pred = eps_neg + cfg_weight * (eps_text - eps_neg)

        x_t_prev = self.sampler.step(eps_pred, x_t, t, t_prev)

        return x_t_prev

    def _denoising_loop(
        self,
        x_T,
        T,
        conditioning,
        num_steps: int = 50,
        cfg_weight: float = 7.5,
        text_time=None,
    ):
        x_t = x_T
        for t, t_prev in self.sampler.timesteps(
            num_steps, start_time=T, dtype=self.dtype
        ):
            x_t = self._denoising_step(
                x_t, t, t_prev, conditioning, cfg_weight, text_time
            )
            yield x_t

    def generate_latents(
        self,
        text: str,
        n_images: int = 1,
        num_steps: int = 50,
        cfg_weight: float = 7.5,
        negative_text: str = "",
        latent_size: Tuple[int] = (64, 64),
        seed=None,
    ):
        # Set the PRNG state
        seed = int(time.time()) if seed is None else seed
        mx.random.seed(seed)

        # Get the text conditioning
        conditioning = self._get_text_conditioning(
            text, n_images, cfg_weight, negative_text
        )

        # Create the latent variables
        x_T = self.sampler.sample_prior(
            (n_images, *latent_size, self.autoencoder.latent_channels), dtype=self.dtype
        )

        # Perform the denoising loop
        yield from self._denoising_loop(
            x_T, self.sampler.max_time, conditioning, num_steps, cfg_weight
        )

    def decode(self, x_t):
        x = self.autoencoder.decode(x_t)
        x = mx.clip(x / 2 + 0.5, 0, 1)
        return x
