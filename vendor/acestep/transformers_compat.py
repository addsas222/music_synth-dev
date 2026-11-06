import json
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from torch import nn


def patch_huggingface_hub():
    import huggingface_hub

    if not hasattr(huggingface_hub, "split_torch_state_dict_into_shards"):
        def split_torch_state_dict_into_shards(
            state_dict,
            filename_pattern: str = "model{suffix}.safetensors",
            max_shard_size: str = "10GB",
            weights_name: Optional[str] = None,
        ):
            filename = weights_name or filename_pattern.replace("{suffix}", "")
            tensor_to_filename = {name: filename for name in state_dict.keys()}
            filename_to_tensors = {filename: list(state_dict.keys())}
            total_size = 0
            for value in state_dict.values():
                if hasattr(value, "numel") and hasattr(value, "element_size"):
                    total_size += int(value.numel()) * int(value.element_size())
            return SimpleNamespace(
                filename_to_tensors=filename_to_tensors,
                tensor_to_filename=tensor_to_filename,
                metadata={"total_size": total_size, "max_shard_size": max_shard_size},
                is_sharded=False,
            )

        huggingface_hub.split_torch_state_dict_into_shards = split_torch_state_dict_into_shards

    return huggingface_hub


patch_huggingface_hub()

from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


LOGGER = logging.get_logger(__name__)
ALL_ATTENTION_FUNCTIONS: Dict[str, Any] = {}


def auto_docstring(obj):
    return obj


def can_return_tuple(fn):
    return fn


class FlashAttentionKwargs(TypedDict, total=False):
    pass


class GradientCheckpointingLayer(nn.Module):
    pass


def layer_type_validation(layer_types, num_hidden_layers=None):
    if layer_types is None:
        return
    if not isinstance(layer_types, (list, tuple)):
        raise ValueError("layer_types must be a list or tuple")
    if num_hidden_layers is not None and len(layer_types) != int(num_hidden_layers):
        raise ValueError("layer_types length does not match num_hidden_layers")
    allowed = {"full_attention", "sliding_attention"}
    for item in layer_types:
        if item not in allowed:
            raise ValueError(f"Unsupported layer type: {item}")



def rope_config_validation(config):
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return
    if not isinstance(rope_scaling, dict):
        raise ValueError("rope_scaling must be a dict when provided")


@dataclass
class _CacheLayer:
    keys: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None


class Cache:
    def __init__(self):
        self.layers: Dict[int, _CacheLayer] = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        layer = self.layers.get(int(layer_idx))
        if layer is None or layer.keys is None or layer.values is None:
            layer = _CacheLayer(keys=key_states, values=value_states)
            self.layers[int(layer_idx)] = layer
        else:
            layer.keys = torch.cat([layer.keys, key_states], dim=-2)
            layer.values = torch.cat([layer.values, value_states], dim=-2)
        return layer.keys, layer.values

    def get_seq_length(self):
        if not self.layers:
            return 0
        first_key = min(self.layers.keys())
        layer = self.layers[first_key]
        if layer.keys is None:
            return 0
        return int(layer.keys.shape[-2])


class DynamicCache(Cache):
    pass


class EncoderDecoderCache:
    def __init__(self, self_attention_cache=None, cross_attention_cache=None):
        self.self_attention_cache = self_attention_cache or DynamicCache()
        self.cross_attention_cache = cross_attention_cache or DynamicCache()
        self.is_updated: Dict[int, bool] = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return self.self_attention_cache.update(key_states, value_states, layer_idx, cache_kwargs)

    def get_seq_length(self):
        return self.self_attention_cache.get_seq_length()


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))



def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout:
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.base = float(getattr(config, "rope_theta", 1000000.0))
        self.max_seq_len_cached = int(config.max_position_embeddings)
        dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3Config(PretrainedConfig):
    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if self.sliding_window is not None and i >= self.max_window_layers else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values if use_cache else None,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Qwen3DecoderLayer"]

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()



def _create_causal_mask(attention_mask, input_shape, dtype, device):
    batch_size, seq_len = input_shape
    device_type = device.type if isinstance(device, torch.device) else str(device)
    mask_dtype = dtype
    if dtype == torch.bfloat16 and (device_type.startswith("cuda") or device_type.startswith("xpu")):
        mask_dtype = torch.float32
    min_value = torch.finfo(mask_dtype).min
    mask = torch.full((seq_len, seq_len), min_value, device=device, dtype=mask_dtype)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    if attention_mask is not None:
        if attention_mask.dtype != mask_dtype:
            attention_mask = attention_mask.to(mask_dtype)
        padding_mask = (1.0 - attention_mask[:, None, None, :]) * min_value
        mask = mask + padding_mask
    return mask


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_len = inputs_embeds.shape[:2]
        if cache_position is None:
            cache_position = torch.arange(seq_len, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        causal_mask = _create_causal_mask(attention_mask, (batch_size, seq_len), inputs_embeds.dtype, inputs_embeds.device)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None)



def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)



def _normalize_token_value(value):
    if isinstance(value, dict):
        return value.get("content")
    return value


class Qwen3BPETokenizer:
    def __init__(self, tokenizer, pad_token=None, eos_token=None, unk_token=None, model_max_length=131072):
        self._tokenizer = tokenizer
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.unk_token = unk_token or pad_token
        self.model_max_length = int(model_max_length)
        self.padding_side = "right"
        self.clean_up_tokenization_spaces = False
        self.pad_token_id = tokenizer.token_to_id(self.pad_token) if self.pad_token else None
        self.eos_token_id = tokenizer.token_to_id(self.eos_token) if self.eos_token else None
        self.unk_token_id = tokenizer.token_to_id(self.unk_token) if self.unk_token else None

    def encode(self, text, add_special_tokens=False):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(self, text, padding=False, truncation=False, max_length=None, return_tensors=None):
        texts = [text] if isinstance(text, str) else list(text)
        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=False)
        id_rows = []
        for encoding in encodings:
            ids = list(encoding.ids)
            if truncation and max_length is not None:
                ids = ids[: int(max_length)]
            id_rows.append(ids)

        if padding in (True, "longest", "max_length"):
            if padding == "max_length" and max_length is not None:
                target_len = int(max_length)
            else:
                target_len = max((len(row) for row in id_rows), default=0)
                if max_length is not None:
                    target_len = min(target_len, int(max_length))
            pad_id = self.pad_token_id if self.pad_token_id is not None else 0
            padded_rows = []
            masks = []
            for row in id_rows:
                trimmed = row[:target_len]
                pad_len = max(0, target_len - len(trimmed))
                padded_rows.append(trimmed + [pad_id] * pad_len)
                masks.append([1] * len(trimmed) + [0] * pad_len)
        else:
            padded_rows = id_rows
            masks = [[1] * len(row) for row in id_rows]

        if return_tensors == "pt":
            return SimpleNamespace(
                input_ids=torch.tensor(padded_rows, dtype=torch.long),
                attention_mask=torch.tensor(masks, dtype=torch.long),
            )
        return {"input_ids": padded_rows, "attention_mask": masks}



def load_qwen3_tokenizer(model_dir: str):
    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError as exc:
        raise ImportError("tokenizers is required to load the Qwen3 tokenizer") from exc

    vocab_file = os.path.join(model_dir, "vocab.json")
    merges_file = os.path.join(model_dir, "merges.txt")
    if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
        raise FileNotFoundError(f"Tokenizer vocab/merges not found in {model_dir}")

    tokenizer_config = _load_json(os.path.join(model_dir, "tokenizer_config.json"))
    special_tokens_map = _load_json(os.path.join(model_dir, "special_tokens_map.json"))
    added_tokens = _load_json(os.path.join(model_dir, "added_tokens.json"))

    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    ordered_added_tokens = [token for token, _ in sorted(added_tokens.items(), key=lambda item: item[1])]
    if ordered_added_tokens:
        tokenizer.add_special_tokens(ordered_added_tokens)

    pad_token = _normalize_token_value(special_tokens_map.get("pad_token", tokenizer_config.get("pad_token")))
    eos_token = _normalize_token_value(special_tokens_map.get("eos_token", tokenizer_config.get("eos_token")))
    unk_token = _normalize_token_value(special_tokens_map.get("unk_token", tokenizer_config.get("unk_token")))

    wrapper = Qwen3BPETokenizer(
        tokenizer=tokenizer,
        pad_token=pad_token,
        eos_token=eos_token,
        unk_token=unk_token,
        model_max_length=tokenizer_config.get("model_max_length", 131072),
    )
    wrapper.clean_up_tokenization_spaces = bool(tokenizer_config.get("clean_up_tokenization_spaces", False))
    wrapper.padding_side = tokenizer_config.get("padding_side", "right")
    return wrapper



def _load_state_dict(model_dir: str):
    safetensor_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(safetensor_path):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required to load Qwen3 text encoder weights") from exc
        return load_file(safetensor_path)

    for candidate in ("pytorch_model.bin", "model.bin"):
        candidate_path = os.path.join(model_dir, candidate)
        if os.path.exists(candidate_path):
            return torch.load(candidate_path, map_location="cpu")

    raise FileNotFoundError(f"No supported Qwen3 weight file found in {model_dir}")



def load_qwen3_text_encoder(model_dir: str):
    config = Qwen3Config.from_pretrained(model_dir)
    model = Qwen3Model(config)
    state_dict = _load_state_dict(model_dir)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        LOGGER.warning(f"Qwen3 text encoder missing keys: {missing_keys[:10]}")
    if unexpected_keys:
        LOGGER.warning(f"Qwen3 text encoder unexpected keys: {unexpected_keys[:10]}")
    model.eval()
    return model


__all__ = [
    "ALL_ATTENTION_FUNCTIONS",
    "BaseModelOutput",
    "BaseModelOutputWithPast",
    "Cache",
    "DynamicCache",
    "EncoderDecoderCache",
    "FlashAttentionKwargs",
    "GradientCheckpointingLayer",
    "PreTrainedModel",
    "PretrainedConfig",
    "Qwen3Config",
    "Qwen3MLP",
    "Qwen3Model",
    "Qwen3RMSNorm",
    "Qwen3RotaryEmbedding",
    "Unpack",
    "apply_rotary_pos_emb",
    "auto_docstring",
    "can_return_tuple",
    "eager_attention_forward",
    "layer_type_validation",
    "load_qwen3_text_encoder",
    "load_qwen3_tokenizer",
    "logging",
    "patch_huggingface_hub",
    "rope_config_validation",
]
