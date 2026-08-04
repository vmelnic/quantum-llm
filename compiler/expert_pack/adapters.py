"""Explicit, fail-closed architecture adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import AdapterError
from .safetensors import SafeTensorCheckpoint, TensorInfo

EXPERT_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class ExpertSource:
    layer: int
    expert: int
    gate: TensorInfo
    up: TensorInfo
    down: TensorInfo


@dataclass(frozen=True)
class AdaptedModel:
    family: str
    architecture: dict[str, object]
    dense: tuple[TensorInfo, ...]
    experts: tuple[ExpertSource, ...]
    source_tensor_count: int


def _integer(config: dict[str, object], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdapterError(f"model config requires positive integer {key}")
    return value


class OlmoeAdapter:
    name = "olmoe"

    def adapt(self, checkpoint: SafeTensorCheckpoint) -> AdaptedModel:
        config = checkpoint.config
        if config.get("model_type") != "olmoe":
            raise AdapterError("olmoe adapter requires config.model_type == 'olmoe'")
        architectures = config.get("architectures")
        if architectures is not None and (
            not isinstance(architectures, list) or "OlmoeForCausalLM" not in architectures
        ):
            raise AdapterError("olmoe adapter requires OlmoeForCausalLM architecture")

        layers = _integer(config, "num_hidden_layers")
        experts = _integer(config, "num_experts")
        top_k = _integer(config, "num_experts_per_tok")
        hidden = _integer(config, "hidden_size")
        intermediate = _integer(config, "intermediate_size")
        vocab = _integer(config, "vocab_size")
        heads = _integer(config, "num_attention_heads")
        kv_heads = _integer(config, "num_key_value_heads")
        max_context = _integer(config, "max_position_embeddings")
        if hidden % heads:
            raise AdapterError("hidden_size is not divisible by num_attention_heads")
        if top_k > experts:
            raise AdapterError("num_experts_per_tok exceeds num_experts")
        head_dim = hidden // heads
        clip_qkv = config.get("clip_qkv")
        if clip_qkv is not None and (
            isinstance(clip_qkv, bool)
            or not isinstance(clip_qkv, (int, float))
            or float(clip_qkv) <= 0.0
        ):
            raise AdapterError("clip_qkv must be null or a positive number")

        expected_dense: dict[str, tuple[int, ...]] = {
            "model.embed_tokens.weight": (vocab, hidden),
            "model.norm.weight": (hidden,),
        }
        if not bool(config.get("tie_word_embeddings", False)):
            expected_dense["lm_head.weight"] = (vocab, hidden)
        for layer in range(layers):
            prefix = f"model.layers.{layer}."
            expected_dense.update(
                {
                    prefix + "input_layernorm.weight": (hidden,),
                    prefix + "post_attention_layernorm.weight": (hidden,),
                    prefix + "self_attn.q_proj.weight": (hidden, hidden),
                    prefix + "self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
                    prefix + "self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
                    prefix + "self_attn.o_proj.weight": (hidden, hidden),
                    # OLMoE applies q/k norms to the full projected hidden
                    # vector; the checkpoint tensors are [hidden_size], not
                    # one shared [head_dim] vector.
                    prefix + "self_attn.q_norm.weight": (hidden,),
                    prefix + "self_attn.k_norm.weight": (hidden,),
                    prefix + "mlp.gate.weight": (experts, hidden),
                }
            )

        routed: dict[tuple[int, int], dict[str, TensorInfo]] = {}
        dense_by_name: dict[str, TensorInfo] = {}
        unknown: list[str] = []
        for name, info in checkpoint.tensors.items():
            match = EXPERT_PATTERN.fullmatch(name)
            if match:
                key = (int(match["layer"]), int(match["expert"]))
                projection = match["projection"]
                if key[0] >= layers or key[1] >= experts:
                    raise AdapterError(f"expert tensor outside configured bounds: {name}")
                routed.setdefault(key, {})[projection] = info
            elif name in expected_dense:
                dense_by_name[name] = info
            else:
                unknown.append(name)
        if unknown:
            raise AdapterError(f"unidentified OLMoE tensors: {sorted(unknown)[:8]}")

        missing_dense = sorted(set(expected_dense) - set(dense_by_name))
        if missing_dense:
            raise AdapterError(f"missing OLMoE dense tensors: {missing_dense[:8]}")
        for name, expected_shape in expected_dense.items():
            actual = dense_by_name[name].shape
            if actual != expected_shape:
                raise AdapterError(
                    f"dense shape mismatch for {name}: expected {expected_shape}, got {actual}"
                )

        expert_sources: list[ExpertSource] = []
        for layer in range(layers):
            for expert in range(experts):
                projections = routed.get((layer, expert), {})
                if set(projections) != {"gate_proj", "up_proj", "down_proj"}:
                    raise AdapterError(
                        f"expert ({layer}, {expert}) projections are "
                        f"{sorted(projections)}, expected gate/up/down"
                    )
                gate = projections["gate_proj"]
                up = projections["up_proj"]
                down = projections["down_proj"]
                for projection, info, shape in (
                    ("gate", gate, (intermediate, hidden)),
                    ("up", up, (intermediate, hidden)),
                    ("down", down, (hidden, intermediate)),
                ):
                    if info.shape != shape:
                        raise AdapterError(
                            f"expert ({layer}, {expert}) {projection} shape mismatch: "
                            f"expected {shape}, got {info.shape}"
                        )
                expert_sources.append(ExpertSource(layer, expert, gate, up, down))

        if len(dense_by_name) + len(expert_sources) * 3 != len(checkpoint.tensors):
            raise AdapterError("adapter classification is not a one-to-one tensor partition")

        architecture = {
            "family": self.name,
            "model_type": "olmoe",
            "architectures": architectures or ["OlmoeForCausalLM"],
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "vocab_size": vocab,
            "max_position_embeddings": max_context,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "num_experts": experts,
            "num_experts_per_token": top_k,
            "shared_experts": 0,
            "hidden_activation": config.get("hidden_act", "silu"),
            "attention_bias": bool(config.get("attention_bias", False)),
            "attention_dropout": config.get("attention_dropout", 0.0),
            "clip_qkv": clip_qkv,
            "rms_norm_epsilon": config.get("rms_norm_eps"),
            "rope": {
                "theta": config.get("rope_theta", 10000.0),
                "scaling": config.get("rope_scaling"),
            },
            "normalize_topk_probability": bool(config.get("norm_topk_prob", False)),
            "tie_word_embeddings": bool(config.get("tie_word_embeddings", False)),
        }
        return AdaptedModel(
            family=self.name,
            architecture=architecture,
            dense=tuple(dense_by_name[name] for name in sorted(dense_by_name)),
            experts=tuple(expert_sources),
            source_tensor_count=len(checkpoint.tensors),
        )


class Qwen3NextAdapter:
    name = "qwen3_next"

    def adapt(self, checkpoint: SafeTensorCheckpoint) -> AdaptedModel:
        config = checkpoint.config
        if config.get("model_type") != "qwen3_next":
            raise AdapterError("qwen3_next adapter requires config.model_type == 'qwen3_next'")
        architectures = config.get("architectures")
        if architectures is not None and (
            not isinstance(architectures, list) or "Qwen3NextForCausalLM" not in architectures
        ):
            raise AdapterError("qwen3_next adapter requires Qwen3NextForCausalLM")
        layers = _integer(config, "num_hidden_layers")
        experts = _integer(config, "num_experts")
        top_k = _integer(config, "num_experts_per_tok")
        hidden = _integer(config, "hidden_size")
        expert_width = _integer(config, "moe_intermediate_size")
        shared_width = _integer(config, "shared_expert_intermediate_size")
        vocab = _integer(config, "vocab_size")
        heads = _integer(config, "num_attention_heads")
        kv_heads = _integer(config, "num_key_value_heads")
        head_dim = _integer(config, "head_dim")
        max_context = _integer(config, "max_position_embeddings")
        full_interval = _integer(config, "full_attention_interval")
        conv_kernel = _integer(config, "linear_conv_kernel_dim")
        key_head_dim = _integer(config, "linear_key_head_dim")
        value_head_dim = _integer(config, "linear_value_head_dim")
        key_heads = _integer(config, "linear_num_key_heads")
        value_heads = _integer(config, "linear_num_value_heads")
        if top_k > experts or value_heads % key_heads:
            raise AdapterError("invalid Qwen3-Next expert or linear-head geometry")
        key_dim, value_dim = key_heads * key_head_dim, value_heads * value_head_dim
        conv_dim = 2 * key_dim + value_dim

        expected_dense: dict[str, tuple[int, ...]] = {
            "model.embed_tokens.weight": (vocab, hidden),
            "model.norm.weight": (hidden,),
        }
        if not bool(config.get("tie_word_embeddings", False)):
            expected_dense["lm_head.weight"] = (vocab, hidden)
        for layer in range(layers):
            prefix = f"model.layers.{layer}."
            expected_dense.update({
                prefix + "input_layernorm.weight": (hidden,),
                prefix + "post_attention_layernorm.weight": (hidden,),
                prefix + "mlp.gate.weight": (experts, hidden),
                prefix + "mlp.shared_expert.gate_proj.weight": (shared_width, hidden),
                prefix + "mlp.shared_expert.up_proj.weight": (shared_width, hidden),
                prefix + "mlp.shared_expert.down_proj.weight": (hidden, shared_width),
                prefix + "mlp.shared_expert_gate.weight": (1, hidden),
            })
            if (layer + 1) % full_interval == 0:
                expected_dense.update({
                    # Qwen3-Next projects query and an attention output gate
                    # together, then chunks the last dimension in half.
                    prefix + "self_attn.q_proj.weight": (2 * heads * head_dim, hidden),
                    prefix + "self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
                    prefix + "self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
                    prefix + "self_attn.o_proj.weight": (hidden, heads * head_dim),
                    prefix + "self_attn.q_norm.weight": (head_dim,),
                    prefix + "self_attn.k_norm.weight": (head_dim,),
                })
            else:
                expected_dense.update({
                    prefix + "linear_attn.dt_bias": (value_heads,),
                    prefix + "linear_attn.A_log": (value_heads,),
                    prefix + "linear_attn.conv1d.weight": (conv_dim, 1, conv_kernel),
                    prefix + "linear_attn.in_proj_qkvz.weight": (2 * key_dim + 2 * value_dim, hidden),
                    prefix + "linear_attn.in_proj_ba.weight": (2 * value_heads, hidden),
                    prefix + "linear_attn.norm.weight": (value_head_dim,),
                    prefix + "linear_attn.out_proj.weight": (hidden, value_dim),
                })

        # The official Instruct checkpoint also carries one multi-token
        # prediction (MTP) decoder. It is not needed for ordinary
        # autoregressive decoding, but Expert Pack v1 preserves every source
        # tensor. Store this optional auxiliary head in dense.qpack under its
        # original names so runtimes may ignore or opt into it explicitly.
        has_mtp = any(name.startswith("mtp.") for name in checkpoint.tensors)
        if has_mtp:
            expected_dense.update({
                "mtp.fc.weight": (hidden, 2 * hidden),
                "mtp.pre_fc_norm_embedding.weight": (hidden,),
                "mtp.pre_fc_norm_hidden.weight": (hidden,),
                "mtp.norm.weight": (hidden,),
            })
            prefix = "mtp.layers.0."
            expected_dense.update({
                prefix + "input_layernorm.weight": (hidden,),
                prefix + "post_attention_layernorm.weight": (hidden,),
                prefix + "self_attn.q_proj.weight": (2 * heads * head_dim, hidden),
                prefix + "self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
                prefix + "self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
                prefix + "self_attn.o_proj.weight": (hidden, heads * head_dim),
                prefix + "self_attn.q_norm.weight": (head_dim,),
                prefix + "self_attn.k_norm.weight": (head_dim,),
                prefix + "mlp.gate.weight": (experts, hidden),
                prefix + "mlp.shared_expert.gate_proj.weight": (shared_width, hidden),
                prefix + "mlp.shared_expert.up_proj.weight": (shared_width, hidden),
                prefix + "mlp.shared_expert.down_proj.weight": (hidden, shared_width),
                prefix + "mlp.shared_expert_gate.weight": (1, hidden),
            })
            for expert in range(experts):
                expert_prefix = prefix + f"mlp.experts.{expert}."
                expected_dense.update({
                    expert_prefix + "gate_proj.weight": (expert_width, hidden),
                    expert_prefix + "up_proj.weight": (expert_width, hidden),
                    expert_prefix + "down_proj.weight": (hidden, expert_width),
                })

        routed: dict[tuple[int, int], dict[str, TensorInfo]] = {}
        dense_by_name: dict[str, TensorInfo] = {}
        unknown: list[str] = []
        for name, info in checkpoint.tensors.items():
            match = EXPERT_PATTERN.fullmatch(name)
            if match:
                key = (int(match["layer"]), int(match["expert"]))
                if key[0] >= layers or key[1] >= experts:
                    raise AdapterError(f"expert tensor outside configured bounds: {name}")
                routed.setdefault(key, {})[match["projection"]] = info
            elif name in expected_dense:
                dense_by_name[name] = info
            else:
                unknown.append(name)
        if unknown:
            raise AdapterError(f"unidentified Qwen3-Next tensors: {sorted(unknown)[:8]}")
        missing = sorted(set(expected_dense) - set(dense_by_name))
        if missing:
            raise AdapterError(f"missing Qwen3-Next dense tensors: {missing[:8]}")
        for name, shape in expected_dense.items():
            if dense_by_name[name].shape != shape:
                raise AdapterError(
                    f"dense shape mismatch for {name}: expected {shape}, got {dense_by_name[name].shape}"
                )
        sources: list[ExpertSource] = []
        for layer in range(layers):
            for expert in range(experts):
                projections = routed.get((layer, expert), {})
                if set(projections) != {"gate_proj", "up_proj", "down_proj"}:
                    raise AdapterError(f"expert ({layer}, {expert}) is incomplete")
                gate, up, down = (projections[name] for name in ("gate_proj", "up_proj", "down_proj"))
                for label, info, shape in (
                    ("gate", gate, (expert_width, hidden)),
                    ("up", up, (expert_width, hidden)),
                    ("down", down, (hidden, expert_width)),
                ):
                    if info.shape != shape:
                        raise AdapterError(
                            f"expert ({layer}, {expert}) {label} shape mismatch: expected {shape}, got {info.shape}"
                        )
                sources.append(ExpertSource(layer, expert, gate, up, down))
        if len(dense_by_name) + len(sources) * 3 != len(checkpoint.tensors):
            raise AdapterError("Qwen3-Next classification is not a one-to-one tensor partition")
        partial_rotary = config.get("partial_rotary_factor")
        if not isinstance(partial_rotary, (int, float)) or not 0 < float(partial_rotary) <= 1:
            raise AdapterError("partial_rotary_factor must be in (0,1]")
        architecture = {
            "family": self.name,
            "model_type": self.name,
            "architectures": architectures or ["Qwen3NextForCausalLM"],
            "hidden_size": hidden,
            "intermediate_size": expert_width,
            "vocab_size": vocab,
            "max_position_embeddings": max_context,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "num_experts": experts,
            "num_experts_per_token": top_k,
            "shared_experts": 1,
            "hidden_activation": config.get("hidden_act", "silu"),
            "attention_bias": bool(config.get("attention_bias", False)),
            "attention_dropout": config.get("attention_dropout", 0.0),
            "clip_qkv": None,
            "rms_norm_epsilon": config.get("rms_norm_eps"),
            "rope": {"theta": config.get("rope_theta", 10000000.0), "scaling": config.get("rope_scaling")},
            "normalize_topk_probability": bool(config.get("norm_topk_prob", True)),
            "tie_word_embeddings": bool(config.get("tie_word_embeddings", False)),
            "full_attention_interval": full_interval,
            "partial_rotary_factor": float(partial_rotary),
            "linear_conv_kernel_dim": conv_kernel,
            "linear_key_head_dim": key_head_dim,
            "linear_value_head_dim": value_head_dim,
            "linear_num_key_heads": key_heads,
            "linear_num_value_heads": value_heads,
            "shared_expert_intermediate_size": shared_width,
            "multi_token_prediction_layers": 1 if has_mtp else 0,
            "attention_output_gate": True,
        }
        return AdaptedModel(self.name, architecture,
                            tuple(dense_by_name[name] for name in sorted(dense_by_name)),
                            tuple(sources), len(checkpoint.tensors))


ADAPTERS = {
    OlmoeAdapter.name: OlmoeAdapter(),
    Qwen3NextAdapter.name: Qwen3NextAdapter(),
}


def adapt_checkpoint(checkpoint: SafeTensorCheckpoint, adapter: str) -> AdaptedModel:
    try:
        implementation = ADAPTERS[adapter]
    except KeyError as error:
        raise AdapterError(f"unknown adapter {adapter!r}; supported: {sorted(ADAPTERS)}") from error
    return implementation.adapt(checkpoint)
