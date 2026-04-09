# Copyright © 2023-2024 Apple Inc.

import json
from typing import Optional

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten

from .clip import CLIPTextModel
from .config import AutoencoderConfig, CLIPTextModelConfig, DiffusionConfig, UNetConfig
from .tokenizer import Tokenizer
from .unet import UNetModel
from .vae import Autoencoder

_DEFAULT_MODEL_ID = "CompVis/stable-diffusion-v1-4"
_DEFAULT_SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"


def map_unet_weights(key, value):
    if "downsamplers" in key:
        key = key.replace("downsamplers.0.conv", "downsample")
        key = key.replace("downsamplers.0.op", "downsample")
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
    if "conv_shortcut" in key and len(value.shape) == 4:
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
        key = key.replace("downsamplers.0.op", "downsample")
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
    if "conv_shortcut" in key and len(value.shape) == 4:
        value = value.squeeze()

    if len(value.shape) == 4:
        value = value.transpose(0, 2, 3, 1)
        value = value.reshape(-1).reshape(value.shape)

    if any(
        s in key
        for s in (
            "query_proj.weight",
            "key_proj.weight",
            "value_proj.weight",
            "out_proj.weight",
        )
    ) and len(value.shape) == 4:
        value = value.squeeze()

    return [(key, value)]


def _flatten(params):
    return [(k, v) for p in params for (k, v) in p]


def _filter_weights_to_model(model, weights):
    flat = tree_flatten(model.parameters())
    keys = set()
    for path, _ in flat:
        if isinstance(path, (list, tuple)):
            key = ".".join(str(p) for p in path)
        else:
            key = str(path)
        keys.add(key)

    filtered = {k: v for k, v in weights.items() if k in keys}
    return filtered or weights


def _load_weights(mapper, model, weights, float16: bool = False):
    dtype = mx.float16 if float16 else mx.float32
    weights = _flatten([mapper(k, v.astype(dtype)) for k, v in weights.items()])
    normalized = []
    for k, v in weights:
        if "conv_shortcut" in k and len(v.shape) == 4:
            v = v.squeeze()
        if any(
            s in k
            for s in (
                "query_proj.weight",
                "key_proj.weight",
                "value_proj.weight",
                "out_proj.weight",
            )
        ) and len(v.shape) == 4:
            v = v.squeeze()
        normalized.append((k, v))
    weights = normalized
    weights = dict(weights)
    weights = _filter_weights_to_model(model, weights)
    weights = list(weights.items())
    model.update(tree_unflatten(weights))


def _load_safetensor_weights(mapper, model, weight_file, float16: bool = False):
    weights = mx.load(weight_file)
    _load_weights(mapper, model, weights, float16)


def _load_json(model_id: str, filename: str):
    with open(hf_hub_download(model_id, filename)) as f:
        return json.load(f)


def build_unet_from_config(model_id: str = _DEFAULT_MODEL_ID):
    config = _load_json(model_id, "unet/config.json")
    n_blocks = len(config["block_out_channels"])
    return UNetModel(
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


def build_text_encoder_from_config(model_id: str = _DEFAULT_MODEL_ID):
    config = _load_json(model_id, "text_encoder/config.json")
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
    return model


def build_autoencoder_from_config(model_id: str = _DEFAULT_MODEL_ID):
    config = _load_json(model_id, "vae/config.json")
    return Autoencoder(
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



def load_unet(model_id: str = _DEFAULT_MODEL_ID, float16: bool = False):
    """Load the stable diffusion UNet from Hugging Face Hub."""
    model = build_unet_from_config(model_id)

    # Download the weights and map them into the model
    unet_weights = "unet/diffusion_pytorch_model.safetensors"
    weight_file = hf_hub_download(model_id, unet_weights)
    _load_safetensor_weights(map_unet_weights, model, weight_file, float16)

    return model


def load_text_encoder(model_id: str = _DEFAULT_MODEL_ID, float16: bool = False):
    """Load the stable diffusion text encoder from Hugging Face Hub."""
    model = build_text_encoder_from_config(model_id)

    # Download the weights and map them into the model
    text_encoder_weights = "text_encoder/model.safetensors"
    weight_file = hf_hub_download(model_id, text_encoder_weights)
    _load_safetensor_weights(map_clip_text_encoder_weights, model, weight_file, float16)

    return model


def load_autoencoder(model_id: str = _DEFAULT_MODEL_ID, float16: bool = False):
    """Load the stable diffusion autoencoder from Hugging Face Hub."""
    model = build_autoencoder_from_config(model_id)

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


def _replace_resnet_keys(key: str) -> str:
    key = key.replace("in_layers.0", "norm1")
    key = key.replace("in_layers.2", "conv1")
    key = key.replace("out_layers.0", "norm2")
    key = key.replace("out_layers.3", "conv2")
    key = key.replace("emb_layers.1", "time_emb_proj")
    key = key.replace("skip_connection", "conv_shortcut")
    key = key.replace("nin_shortcut", "conv_shortcut")
    return key


def _replace_attn_keys_unet(key: str) -> str:
    key = key.replace(".attn1.q.", ".attn1.to_q.")
    key = key.replace(".attn1.k.", ".attn1.to_k.")
    key = key.replace(".attn1.v.", ".attn1.to_v.")
    key = key.replace(".attn1.proj_out.", ".attn1.to_out.0.")
    key = key.replace(".attn2.q.", ".attn2.to_q.")
    key = key.replace(".attn2.k.", ".attn2.to_k.")
    key = key.replace(".attn2.v.", ".attn2.to_v.")
    key = key.replace(".attn2.proj_out.", ".attn2.to_out.0.")
    return key


def _replace_attn_keys_vae(key: str) -> str:
    key = key.replace(".norm.", ".group_norm.")
    key = key.replace(".q.", ".query.")
    key = key.replace(".k.", ".key.")
    key = key.replace(".v.", ".value.")
    key = key.replace(".proj_out.", ".proj_attn.")
    return key


def _group_blocks(weights, prefix: str):
    blocks = {}
    for k in weights.keys():
        if not k.startswith(prefix):
            continue
        parts = k.split(".")
        if len(parts) < 3:
            continue
        block_idx = int(parts[1])
        sub_idx = parts[2]
        blocks.setdefault(block_idx, {}).setdefault(sub_idx, []).append(k)
    return blocks


def _convert_unet_ldm_to_diffusers(weights):
    input_blocks = _group_blocks(weights, "input_blocks.")
    output_blocks = _group_blocks(weights, "output_blocks.")

    input_block_map = {}
    down_idx = 0
    resnet_idx = 0
    for i in sorted(input_blocks.keys()):
        if i == 0:
            continue
        sub = input_blocks[i]
        is_downsample = any(
            any(".op." in k or ".downsample." in k for k in keys)
            for keys in sub.values()
        )
        if is_downsample:
            input_block_map[i] = {"type": "downsample", "down_idx": down_idx}
            down_idx += 1
            resnet_idx = 0
        else:
            input_block_map[i] = {
                "type": "resnet",
                "down_idx": down_idx,
                "resnet_idx": resnet_idx,
            }
            resnet_idx += 1

    output_block_map = {}
    up_idx = 0
    resnet_idx = 0
    for i in sorted(output_blocks.keys()):
        sub = output_blocks[i]
        resnet_sub = None
        attn_sub = None
        upsample_sub = None

        for sub_idx, keys in sub.items():
            joined = " ".join(keys)
            if any(
                s in joined
                for s in ("in_layers", "out_layers", "emb_layers", "skip_connection")
            ):
                resnet_sub = sub_idx
            if any(s in joined for s in ("transformer_blocks", ".attn", "proj_in", "proj_out")):
                attn_sub = sub_idx
            if any(
                ".conv." in k and all(
                    s not in k
                    for s in ("in_layers", "out_layers", "emb_layers", "skip_connection")
                )
                for k in keys
            ):
                upsample_sub = sub_idx

        output_block_map[i] = {
            "up_idx": up_idx,
            "resnet_idx": resnet_idx,
            "resnet_sub": resnet_sub,
            "attn_sub": attn_sub,
            "upsample_sub": upsample_sub,
        }
        if upsample_sub is not None:
            up_idx += 1
            resnet_idx = 0
        else:
            resnet_idx += 1

    converted = {}
    for k, v in weights.items():
        if k.startswith("time_embed.0."):
            new_k = k.replace("time_embed.0.", "time_embedding.linear_1.")
        elif k.startswith("time_embed.2."):
            new_k = k.replace("time_embed.2.", "time_embedding.linear_2.")
        elif k.startswith("input_blocks.0.0."):
            new_k = k.replace("input_blocks.0.0.", "conv_in.")
        elif k.startswith("input_blocks."):
            parts = k.split(".")
            block_idx = int(parts[1])
            sub_idx = parts[2]
            rest = ".".join(parts[3:])
            if block_idx == 0:
                continue
            info = input_block_map.get(block_idx)
            if info is None:
                continue
            if info["type"] == "downsample":
                new_k = f"down_blocks.{info['down_idx']}.downsamplers.0.{rest}"
            else:
                if sub_idx == "0":
                    new_k = (
                        f"down_blocks.{info['down_idx']}.resnets.{info['resnet_idx']}."
                        f"{_replace_resnet_keys(rest)}"
                    )
                elif sub_idx == "1":
                    new_k = (
                        f"down_blocks.{info['down_idx']}.attentions.{info['resnet_idx']}."
                        f"{rest}"
                    )
                else:
                    continue
        elif k.startswith("middle_block."):
            parts = k.split(".")
            block_idx = parts[1]
            rest = ".".join(parts[2:])
            if block_idx == "0":
                new_k = f"mid_block.resnets.0.{_replace_resnet_keys(rest)}"
            elif block_idx == "1":
                new_k = f"mid_block.attentions.0.{rest}"
            elif block_idx == "2":
                new_k = f"mid_block.resnets.1.{_replace_resnet_keys(rest)}"
            else:
                continue
        elif k.startswith("output_blocks."):
            parts = k.split(".")
            block_idx = int(parts[1])
            sub_idx = parts[2]
            rest = ".".join(parts[3:])
            info = output_block_map.get(block_idx)
            if info is None:
                continue
            if sub_idx == info["resnet_sub"]:
                new_k = (
                    f"up_blocks.{info['up_idx']}.resnets.{info['resnet_idx']}."
                    f"{_replace_resnet_keys(rest)}"
                )
            elif sub_idx == info["attn_sub"]:
                new_k = (
                    f"up_blocks.{info['up_idx']}.attentions.{info['resnet_idx']}."
                    f"{rest}"
                )
            elif sub_idx == info["upsample_sub"]:
                new_k = f"up_blocks.{info['up_idx']}.upsamplers.0.{rest}"
            else:
                continue
        elif k.startswith("out.0."):
            new_k = k.replace("out.0.", "conv_norm_out.")
        elif k.startswith("out.2."):
            new_k = k.replace("out.2.", "conv_out.")
        else:
            continue

        new_k = _replace_attn_keys_unet(new_k)
        converted[new_k] = v

    return converted


def _convert_vae_ldm_to_diffusers(weights):
    converted = {}
    for k, v in weights.items():
        if k.startswith("encoder.conv_in."):
            new_k = k
        elif k.startswith("encoder.conv_out."):
            new_k = k
        elif k.startswith("encoder.norm_out."):
            new_k = k.replace("encoder.norm_out.", "encoder.conv_norm_out.")
        elif k.startswith("decoder.conv_in."):
            new_k = k
        elif k.startswith("decoder.conv_out."):
            new_k = k
        elif k.startswith("decoder.norm_out."):
            new_k = k.replace("decoder.norm_out.", "decoder.conv_norm_out.")
        elif k.startswith("encoder.down."):
            new_k = k.replace("encoder.down.", "encoder.down_blocks.")
            new_k = new_k.replace(".block.", ".resnets.")
            new_k = new_k.replace(".attn.", ".attentions.")
            new_k = new_k.replace(".downsample.", ".downsamplers.0.")
            new_k = _replace_resnet_keys(new_k)
            new_k = _replace_attn_keys_vae(new_k)
        elif k.startswith("decoder.up."):
            new_k = k.replace("decoder.up.", "decoder.up_blocks.")
            new_k = new_k.replace(".block.", ".resnets.")
            new_k = new_k.replace(".attn.", ".attentions.")
            new_k = new_k.replace(".upsample.", ".upsamplers.0.")
            new_k = _replace_resnet_keys(new_k)
            new_k = _replace_attn_keys_vae(new_k)
        elif k.startswith("encoder.mid."):
            new_k = k.replace("encoder.mid.block_1.", "encoder.mid_block.resnets.0.")
            new_k = new_k.replace("encoder.mid.attn_1.", "encoder.mid_block.attentions.0.")
            new_k = new_k.replace("encoder.mid.block_2.", "encoder.mid_block.resnets.1.")
            new_k = _replace_resnet_keys(new_k)
            new_k = _replace_attn_keys_vae(new_k)
        elif k.startswith("decoder.mid."):
            new_k = k.replace("decoder.mid.block_1.", "decoder.mid_block.resnets.0.")
            new_k = new_k.replace("decoder.mid.attn_1.", "decoder.mid_block.attentions.0.")
            new_k = new_k.replace("decoder.mid.block_2.", "decoder.mid_block.resnets.1.")
            new_k = _replace_resnet_keys(new_k)
            new_k = _replace_attn_keys_vae(new_k)
        elif k.startswith("quant_conv."):
            new_k = k
        elif k.startswith("post_quant_conv."):
            new_k = k.replace("post_quant_conv.", "post_quant_conv.")
        else:
            continue

        converted[new_k] = v

    return converted


def _convert_text_encoder_ldm_to_hf(weights):
    converted = {}
    for k, v in weights.items():
        if k.startswith("transformer."):
            new_k = k.replace("transformer.", "text_model.")
        else:
            new_k = k
        converted[new_k] = v
    return converted


def load_models_from_safetensors(
    ckpt_path: str,
    base_model_id: str = _DEFAULT_SD15_MODEL_ID,
    float16: bool = False,
    use_vae_from_checkpoint: bool = False,
):
    weights = mx.load(ckpt_path)

    unet_weights = {
        k[len("model.diffusion_model.") :]: v
        for k, v in weights.items()
        if k.startswith("model.diffusion_model.")
    }
    if not unet_weights:
        unet_weights = {
            k[len("model_ema.diffusion_model.") :]: v
            for k, v in weights.items()
            if k.startswith("model_ema.diffusion_model.")
        }
    vae_weights = {
        k[len("first_stage_model.") :]: v
        for k, v in weights.items()
        if k.startswith("first_stage_model.")
    }
    text_weights = {}
    for k, v in weights.items():
        if k.startswith("cond_stage_model.transformer."):
            text_weights[k[len("cond_stage_model.transformer.") :]] = v
        elif k.startswith("cond_stage_model."):
            text_weights[k[len("cond_stage_model.") :]] = v

    if not unet_weights or not text_weights:
        raise ValueError(
            "Checkpoint is missing UNet or text encoder weights; only SD1.x "
            "checkpoints are supported."
        )

    unet = build_unet_from_config(base_model_id)
    text_encoder = build_text_encoder_from_config(base_model_id)
    if use_vae_from_checkpoint:
        autoencoder = build_autoencoder_from_config(base_model_id)
    else:
        autoencoder = load_autoencoder(base_model_id, False)

    unet_converted = _convert_unet_ldm_to_diffusers(unet_weights)
    vae_converted = _convert_vae_ldm_to_diffusers(vae_weights) if use_vae_from_checkpoint else None
    text_converted = _convert_text_encoder_ldm_to_hf(text_weights)

    _load_weights(map_unet_weights, unet, unet_converted, float16)
    _load_weights(map_clip_text_encoder_weights, text_encoder, text_converted, float16)
    if use_vae_from_checkpoint and vae_converted:
        _load_weights(map_vae_weights, autoencoder, vae_converted, False)

    return unet, text_encoder, autoencoder
