# Copyright © 2023-2024 Apple Inc.

import json
from typing import Optional

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx.utils import tree_unflatten

from .clip import CLIPTextModel
from .config import AutoencoderConfig, CLIPTextModelConfig, DiffusionConfig, UNetConfig
from .tokenizer import Tokenizer
from .unet import UNetModel
from .vae import Autoencoder

_DEFAULT_MODEL_ID = "CompVis/stable-diffusion-v1-4"


def map_unet_weights(key, value):
    if "downsamplers" in key:
        key = key.replace("downsamplers.0.conv", "downsample")
    if "upsamplers" in key:
        key = key.replace("upsamplers.0.conv", "upsample")

    # Map the mid block
    if "mid_block.resnets.0" in key:
        key = key.replace("mid_block.resnets.0", "mid_blocks.0")
    if "mid_block.attentions.0" in key:
        key = key.replace("mid_block.attentions.0", "mid_blocks.1")
    if "mid_block.resnets.1" in key:
        key = key.replace("mid_block.resnets.1", "mid_blocks.2")

    # Map attention layers
    if "to_k" in key:
        key = key.replace("to_k", "key_proj")
    if "to_out.0" in key:
        key = key.replace("to_out.0", "out_proj")
    if "to_q" in key:
        key = key.replace("to_q", "query_proj")
    if "to_v" in key:
        key = key.replace("to_v", "value_proj")

    # Map transformer ffn
    if "ff.net.2" in key:
        key = key.replace("ff.net.2", "linear3")
    if "ff.net.0" in key:
        k1 = key.replace("ff.net.0.proj", "linear1")
        k2 = key.replace("ff.net.0.proj", "linear2")
        v1, v2 = mx.split(value, 2)

        return [(k1, v1), (k2, v2)]

    if "conv_shortcut.weight" in key:
        value = value.squeeze()

    # Transform the weights from 1x1 convs to linear
    if len(value.shape) == 4 and ("proj_in" in key or "proj_out" in key):
        value = value.squeeze()

    if len(value.shape) == 4:
        value = value.transpose(0, 2, 3, 1)
        value = value.reshape(-1).reshape(value.shape)

    return [(key, value)]


def map_clip_text_encoder_weights(key, value):
    # Remove text_model prefix
    if key.startswith("text_model."):
        key = key[11:]

    # The position_ids are not parameters we want to load
    if key == "embeddings.position_ids":
        return []

    # For the embeddings, the key is already correct (e.g. embeddings.token_embedding.weight)
    # For the encoder layers, we need to remove the "encoder." prefix
    if key.startswith("encoder."):
        key = key[8:]

    # Map attention layers
    if "self_attn." in key:
        key = key.replace("self_attn.", "attention.")
    if "q_proj." in key:
        key = key.replace("q_proj.", "query_proj.")
    if "k_proj." in key:
        key = key.replace("k_proj.", "key_proj.")
    if "v_proj." in key:
        key = key.replace("v_proj.", "value_proj.")

    # Map ffn layers
    if "mlp.fc1" in key:
        key = key.replace("mlp.fc1", "linear1")
    if "mlp.fc2" in key:
        key = key.replace("mlp.fc2", "linear2")

    return [(key, value)]


def map_vae_weights(key, value):
    # Map up/downsampling
    if "downsamplers" in key:
        key = key.replace("downsamplers.0.conv", "downsample")
    if "upsamplers" in key:
        key = key.replace("upsamplers.0.conv", "upsample")

    # Map attention layers
    if "mid_block.attentions.0.query" in key:
        key = key.replace("query", "query_proj")
    if "mid_block.attentions.0.key" in key:
        key = key.replace("key", "key_proj")
    if "mid_block.attentions.0.value" in key:
        key = key.replace("value", "value_proj")
    if "mid_block.attentions.0.proj_attn" in key:
        key = key.replace("proj_attn", "out_proj")

    # Map the mid block
    if "mid_block.resnets.0" in key:
        key = key.replace("mid_block.resnets.0", "mid_blocks.0")
    if "mid_block.attentions.0" in key:
        key = key.replace("mid_block.attentions.0", "mid_blocks.1")
    if "mid_block.resnets.1" in key:
        key = key.replace("mid_block.resnets.1", "mid_blocks.2")

    # Map the quant/post_quant layers
    if "quant_conv" in key:
        key = key.replace("quant_conv", "quant_proj")
        value = value.squeeze()

    # Map the conv_shortcut to linear
    if "conv_shortcut.weight" in key:
        value = value.squeeze()

    if len(value.shape) == 4:
        value = value.transpose(0, 2, 3, 1)
        value = value.reshape(-1).reshape(value.shape)

    return [(key, value)]


def _flatten(params):
    return [(k, v) for p in params for (k, v) in p]


def _load_safetensor_weights(mapper, model, weight_file, float16: bool = False):
    dtype = mx.float16 if float16 else mx.float32
    weights = mx.load(weight_file)
    weights = _flatten([mapper(k, v.astype(dtype)) for k, v in weights.items()])
    model.update(tree_unflatten(weights))





def load_unet(model_id: str = _DEFAULT_MODEL_ID, float16: bool = False):
    """Load the stable diffusion UNet from Hugging Face Hub."""
    # Download the config and create the model
    unet_config = "unet/config.json"
    with open(hf_hub_download(model_id, unet_config)) as f:
        config = json.load(f)

    n_blocks = len(config["block_out_channels"])
    model = UNetModel(
        UNetConfig(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=[config["layers_per_block"]] * n_blocks,
            transformer_layers_per_block=config.get(
                "transformer_layers_per_block", (1,) * 4
            ),
            num_attention_heads=(
                [config["attention_head_dim"]] * n_blocks
                if isinstance(config["attention_head_dim"], int)
                else config["attention_head_dim"]
            ),
            cross_attention_dim=[config["cross_attention_dim"]] * n_blocks,
            norm_num_groups=config.get("norm_num_groups", 32),
            down_block_types=config["down_block_types"],
            up_block_types=config["up_block_types"][::-1],
            addition_embed_type=config.get("addition_embed_type", None),
            addition_time_embed_dim=config.get("addition_time_embed_dim", None),
            projection_class_embeddings_input_dim=config.get(
                "projection_class_embeddings_input_dim", None
            ),
        )
    )

    # Download the weights and map them into the model
    unet_weights = "unet/diffusion_pytorch_model.safetensors"
    weight_file = hf_hub_download(model_id, unet_weights)
    _load_safetensor_weights(map_unet_weights, model, weight_file, float16)

    return model


def load_text_encoder(model_id: str = _DEFAULT_MODEL_ID, float16: bool = False):
    """Load the stable diffusion text encoder from Hugging Face Hub."""
    # Download the config and create the model
    text_encoder_config = "text_encoder/config.json"
    with open(hf_hub_download(model_id, text_encoder_config)) as f:
        config = json.load(f)

    with_projection = "WithProjection" in config["architectures"][0]

    model = CLIPTextModel(
        CLIPTextModelConfig(
            num_layers=config["num_hidden_layers"],
            model_dims=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            max_length=config["max_position_embeddings"],
            vocab_size=config["vocab_size"],
            projection_dim=config["projection_dim"] if with_projection else None,
            hidden_act=config.get("hidden_act", "quick_gelu"),
        )
    )
    print("CLIP Text Model initialized")

    # Download the weights and map them into the model
    text_encoder_weights = "text_encoder/model.safetensors"
    weight_file = hf_hub_download(model_id, text_encoder_weights)
    _load_safetensor_weights(map_clip_text_encoder_weights, model, weight_file, float16)

    return model


def load_autoencoder(model_id: str = _DEFAULT_MODEL_ID, float16: bool = False):
    """Load the stable diffusion autoencoder from Hugging Face Hub."""
    # Download the config and create the model
    vae_config = "vae/config.json"
    with open(hf_hub_download(model_id, vae_config)) as f:
        config = json.load(f)

    model = Autoencoder(
        AutoencoderConfig(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            latent_channels_out=2 * config["latent_channels"],
            latent_channels_in=config["latent_channels"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            norm_num_groups=config.get("norm_num_groups", 32),
            scaling_factor=config.get("scaling_factor", 0.18215),
        )
    )

    # Download the weights and map them into the model
    vae_weights = "vae/diffusion_pytorch_model.safetensors"
    weight_file = hf_hub_download(model_id, vae_weights)
    _load_safetensor_weights(map_vae_weights, model, weight_file, float16)

    return model


def load_diffusion_config(model_id: str = _DEFAULT_MODEL_ID):
    """Load the stable diffusion config from Hugging Face Hub."""
    diffusion_config = "scheduler/scheduler_config.json"
    with open(hf_hub_download(model_id, diffusion_config)) as f:
        config = json.load(f)

    return DiffusionConfig(
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        beta_schedule=config["beta_schedule"],
        num_train_steps=config["num_train_timesteps"],
    )


def load_tokenizer(model_id: str = _DEFAULT_MODEL_ID):
    """Load the stable diffusion tokenizer from Hugging Face Hub."""
    vocab_file = hf_hub_download(model_id, "tokenizer/vocab.json")
    with open(vocab_file, encoding="utf-8") as f:
        vocab = json.load(f)

    merges_file = hf_hub_download(model_id, "tokenizer/merges.txt")
    with open(merges_file, encoding="utf-8") as f:
        bpe_merges = f.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
    bpe_merges = [tuple(m.split()) for m in bpe_merges]
    bpe_ranks = dict(map(reversed, enumerate(bpe_merges)))

    return Tokenizer(bpe_ranks, vocab)
