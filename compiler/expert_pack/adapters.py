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
        raise AdapterError(f"OLMoE config requires positive integer {key}")
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


ADAPTERS = {OlmoeAdapter.name: OlmoeAdapter()}


def adapt_checkpoint(checkpoint: SafeTensorCheckpoint, adapter: str) -> AdaptedModel:
    try:
        implementation = ADAPTERS[adapter]
    except KeyError as error:
        raise AdapterError(f"unknown adapter {adapter!r}; supported: {sorted(ADAPTERS)}") from error
    return implementation.adapt(checkpoint)
