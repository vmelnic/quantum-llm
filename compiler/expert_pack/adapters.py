"""Explicit, fail-closed architecture adapters."""

from __future__ import annotations

import re
import math
import struct
from dataclasses import dataclass

from .constants import FP4_QUANT_PROFILE, QUANT_PROFILES
from .errors import AdapterError
from .safetensors import SafeTensorCheckpoint, TensorInfo

EXPERT_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
LFM2_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.feed_forward\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>w1|w2|w3)\.weight$"
)


@dataclass(frozen=True)
class ExpertSource:
    layer: int
    expert: int
    gate: TensorInfo
    up: TensorInfo
    down: TensorInfo


@dataclass(frozen=True)
class RuntimeComponentTopology:
    name: str
    layer_count: int
    experts_per_layer: int
    route_width: int
    shared_experts_per_layer: int
    hidden_size: int
    intermediate_size: int
    execution_capability: str
    router_capability: str
    execution_abi: int = 1
    router_abi: int = 1
    attributes: tuple[tuple[str, int], ...] = ()
    router_parameters: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class RuntimeLayerTopology:
    logical_layer: int
    block_capability: str
    block_abi: int
    routed_component: str | None
    component_layer: int
    parameters: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class RuntimeOperationTopology:
    logical_layer: int | None
    capability: str
    abi: int
    routed_component: str | None
    component_layer: int
    parameters: tuple[tuple[str, int], ...] = ()
    tensor_bindings: tuple[tuple[str, str], ...] = ()
    input_bindings: tuple[tuple[str, str, str], ...] = ()
    output_bindings: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class RuntimeExactDecodeTopology:
    capability: str
    abi: int
    maximum_emitted_tokens: int
    parameters: tuple[tuple[str, int], ...] = ()
    tensor_bindings: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class RuntimeModelTopology:
    architecture_id: str
    vocab_size: int
    max_context_tokens: int
    hidden_size: int
    attributes: tuple[tuple[str, int], ...]
    required_kernels: tuple[tuple[str, int], ...]
    components: tuple[RuntimeComponentTopology, ...]
    layers: tuple[RuntimeLayerTopology, ...]
    operations: tuple[RuntimeOperationTopology, ...]
    tensor_bindings: tuple[tuple[str, str], ...] = ()
    program_inputs: tuple[tuple[str, str, str], ...] = ()
    program_outputs: tuple[tuple[str, str, str], ...] = ()
    exact_decode: RuntimeExactDecodeTopology | None = None


TOKEN_BATCH_ABI = "batch.token-id.u32.host.v1"
POSITION_BATCH_ABI = "batch.position.u32.host.v1"
HIDDEN_BATCH_ABI = "batch.hidden.f32.cuda.v1"
MULTIMODAL_BATCH_ABI = "request.multimodal.fp32.host.v1"
ROUTE_INDEX_BATCH_ABI = "batch.route-index.u32.cuda.v1"
ROUTE_WEIGHT_BATCH_ABI = "batch.route-weight.f32.cuda.v1"


@dataclass(frozen=True)
class AdaptedModel:
    family: str
    architecture: dict[str, object]
    dense: tuple[TensorInfo, ...]
    experts: tuple[ExpertSource, ...]
    source_tensor_count: int
    runtime_topology: RuntimeModelTopology
    # Adapters own semantic tensor roles. The writer must not infer a router
    # from model-family-specific path fragments and accidentally quantize it.
    dense_float32: frozenset[str] = frozenset()
    # Dense FP4 placement is also semantic and adapter-owned. It is applied
    # only when the selected container profile is FP4; existing adapters keep
    # their current INT8/FP32 dense representation by leaving this set empty.
    dense_fp4: frozenset[str] = frozenset()
    supported_expert_quant_profiles: frozenset[str] = frozenset(QUANT_PROFILES)


def _standard_routed_operations(
    layers: tuple[RuntimeLayerTopology, ...],
    components: tuple[RuntimeComponentTopology, ...],
) -> tuple[RuntimeOperationTopology, ...]:
    """Build the common block/router/expert sequence explicitly in an adapter."""
    by_name = {component.name: component for component in components}
    result: list[RuntimeOperationTopology] = []
    for layer in layers:
        result.append(RuntimeOperationTopology(
            logical_layer=layer.logical_layer,
            capability=layer.block_capability,
            abi=layer.block_abi,
            routed_component=None,
            component_layer=0,
            parameters=layer.parameters,
        ))
        if layer.routed_component is None:
            continue
        component = by_name[layer.routed_component]
        result.extend((
            RuntimeOperationTopology(
                logical_layer=layer.logical_layer,
                capability=component.router_capability,
                abi=component.router_abi,
                routed_component=component.name,
                component_layer=layer.component_layer,
            ),
            RuntimeOperationTopology(
                logical_layer=layer.logical_layer,
                capability=component.execution_capability,
                abi=component.execution_abi,
                routed_component=component.name,
                component_layer=layer.component_layer,
            ),
        ))
    return tuple(result)


def _integer(config: dict[str, object], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdapterError(f"model config requires positive integer {key}")
    return value


def _number(config: dict[str, object], key: str) -> float:
    value = config.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise AdapterError(f"model config requires positive finite number {key}")
    return float(value)


def _float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


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
        runtime_component = RuntimeComponentTopology(
            name="decoder",
            layer_count=layers,
            experts_per_layer=experts,
            route_width=top_k,
            shared_experts_per_layer=0,
            hidden_size=hidden,
            intermediate_size=intermediate,
            execution_capability="moe.swiglu.routed.v1",
            router_capability="router.linear.topk.v1",
            router_parameters=(("normalize", int(bool(
                config.get("norm_topk_prob", False)
            ))),),
        )
        norm_bits = _float32_bits(float(config.get("rms_norm_eps", 1e-5)))
        rope_bits = _float32_bits(float(config.get("rope_theta", 10000.0)))
        runtime_layers = tuple(
            RuntimeLayerTopology(
                logical_layer=layer,
                block_capability="block.full-attention.causal.v1",
                block_abi=1,
                routed_component="decoder",
                component_layer=layer,
                parameters=(
                    ("norm_epsilon_f32_bits", norm_bits),
                    ("head_dim", head_dim),
                    ("rope_theta_f32_bits", rope_bits),
                ),
            )
            for layer in range(layers)
        )
        output_head = (
            "model.embed_tokens.weight"
            if bool(config.get("tie_word_embeddings", False))
            else "lm_head.weight"
        )
        runtime_operations: list[RuntimeOperationTopology] = [
            RuntimeOperationTopology(
                logical_layer=None,
                capability="embedding.lookup.int8-row.v1",
                abi=1,
                routed_component=None,
                component_layer=0,
                tensor_bindings=(("weight", "model.embed_tokens.weight"),),
                input_bindings=(("token_ids", "request.token_ids", TOKEN_BATCH_ABI),),
                output_bindings=(("hidden", "hidden.0", HIDDEN_BATCH_ABI),),
            )
        ]
        for layer in range(layers):
            prefix = f"model.layers.{layer}."
            block_hidden = f"layer.{layer}.after_block"
            expert_input = f"layer.{layer}.expert_input"
            route_indices = f"layer.{layer}.route_indices"
            route_weights = f"layer.{layer}.route_weights"
            residual = f"layer.{layer}.residual"
            runtime_operations.extend((
                RuntimeOperationTopology(
                    logical_layer=layer,
                    capability="block.full-attention.causal.v1",
                    abi=1,
                    routed_component=None,
                    component_layer=0,
                    parameters=runtime_layers[layer].parameters,
                    tensor_bindings=(
                        ("input_norm", prefix + "input_layernorm.weight"),
                        ("query_projection", prefix + "self_attn.q_proj.weight"),
                        ("key_projection", prefix + "self_attn.k_proj.weight"),
                        ("value_projection", prefix + "self_attn.v_proj.weight"),
                        ("output_projection", prefix + "self_attn.o_proj.weight"),
                        ("query_norm", prefix + "self_attn.q_norm.weight"),
                        ("key_norm", prefix + "self_attn.k_norm.weight"),
                    ),
                    input_bindings=(
                        ("hidden", f"hidden.{layer}", HIDDEN_BATCH_ABI),
                        ("positions", "request.positions", POSITION_BATCH_ABI),
                    ),
                    output_bindings=(("hidden", block_hidden, HIDDEN_BATCH_ABI),),
                ),
                RuntimeOperationTopology(
                    logical_layer=layer,
                    capability="router.linear.topk.v1",
                    abi=1,
                    routed_component="decoder",
                    component_layer=layer,
                    parameters=(("norm_epsilon_f32_bits", norm_bits),),
                    tensor_bindings=(
                        ("input_norm", prefix + "post_attention_layernorm.weight"),
                        ("router_weight", prefix + "mlp.gate.weight"),
                    ),
                    input_bindings=(("hidden", block_hidden, HIDDEN_BATCH_ABI),),
                    output_bindings=(
                        ("expert_input", expert_input, HIDDEN_BATCH_ABI),
                        ("route_indices", route_indices, ROUTE_INDEX_BATCH_ABI),
                        ("route_weights", route_weights, ROUTE_WEIGHT_BATCH_ABI),
                        ("residual", residual, HIDDEN_BATCH_ABI),
                    ),
                ),
                RuntimeOperationTopology(
                    logical_layer=layer,
                    capability="moe.swiglu.routed.v1",
                    abi=1,
                    routed_component="decoder",
                    component_layer=layer,
                    input_bindings=(
                        ("expert_input", expert_input, HIDDEN_BATCH_ABI),
                        ("route_indices", route_indices, ROUTE_INDEX_BATCH_ABI),
                        ("route_weights", route_weights, ROUTE_WEIGHT_BATCH_ABI),
                        ("residual", residual, HIDDEN_BATCH_ABI),
                    ),
                    output_bindings=(
                        ("hidden", f"hidden.{layer + 1}", HIDDEN_BATCH_ABI),
                    ),
                ),
            ))
        runtime_operations.append(RuntimeOperationTopology(
            logical_layer=None,
            capability="head.rmsnorm.argmax.int8-row.v1",
            abi=1,
            routed_component=None,
            component_layer=0,
            tensor_bindings=(
                ("norm", "model.norm.weight"),
                ("weight", output_head),
            ),
            input_bindings=(("hidden", f"hidden.{layers}", HIDDEN_BATCH_ABI),),
            output_bindings=(("token_ids", "response.token_ids", TOKEN_BATCH_ABI),),
        ))
        runtime_topology = RuntimeModelTopology(
            architecture_id=self.name,
            vocab_size=vocab,
            max_context_tokens=max_context,
            hidden_size=hidden,
            attributes=(
                ("attention_heads", heads),
                ("kv_heads", kv_heads),
                ("head_dim", head_dim),
                ("norm_epsilon_f32_bits", norm_bits),
            ),
            required_kernels=(
                ("embedding.lookup.int8-row.v1", 1),
                ("block.full-attention.causal.v1", 1),
                ("router.linear.topk.v1", 1),
                ("moe.swiglu.routed.v1", 1),
                ("head.rmsnorm.argmax.int8-row.v1", 1),
            ),
            components=(runtime_component,),
            layers=runtime_layers,
            operations=tuple(runtime_operations),
            tensor_bindings=(
                ("token_embedding", "model.embed_tokens.weight"),
                ("final_norm", "model.norm.weight"),
                ("output_head", output_head),
            ),
            program_inputs=(
                ("token_ids", "request.token_ids", TOKEN_BATCH_ABI),
                ("positions", "request.positions", POSITION_BATCH_ABI),
            ),
            program_outputs=(
                ("next_token_ids", "response.token_ids", TOKEN_BATCH_ABI),
            ),
        )
        return AdaptedModel(
            family=self.name,
            architecture=architecture,
            dense=tuple(dense_by_name[name] for name in sorted(dense_by_name)),
            experts=tuple(expert_sources),
            source_tensor_count=len(checkpoint.tensors),
            runtime_topology=runtime_topology,
            dense_float32=frozenset(
                f"model.layers.{layer}.mlp.gate.weight"
                for layer in range(layers)
            ),
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
        runtime_component = RuntimeComponentTopology(
            name="decoder",
            layer_count=layers,
            experts_per_layer=experts,
            route_width=top_k,
            shared_experts_per_layer=1,
            hidden_size=hidden,
            intermediate_size=expert_width,
            execution_capability="moe.swiglu.routed.merge-shared.v1",
            router_capability="router.linear-topk.shared-swiglu.v1",
            router_parameters=(("normalize", int(bool(
                config.get("norm_topk_prob", True)
            ))),),
        )
        runtime_layers = tuple(
            RuntimeLayerTopology(
                logical_layer=layer,
                block_capability=(
                    "block.full-attention.output-gated.v1"
                    if (layer + 1) % full_interval == 0
                    else "block.recurrent-linear-attention.gated-delta.v1"
                ),
                block_abi=1,
                routed_component="decoder",
                component_layer=layer,
            )
            for layer in range(layers)
        )
        norm_bits = _float32_bits(float(config.get("rms_norm_eps", 1e-6)))
        rope_bits = _float32_bits(float(config.get("rope_theta", 10000000.0)))
        rotary_dim = int(head_dim * float(partial_rotary))
        if rotary_dim <= 0 or rotary_dim > head_dim:
            raise AdapterError("Qwen3-Next rotary dimension is invalid")
        output_head = (
            "model.embed_tokens.weight"
            if bool(config.get("tie_word_embeddings", False))
            else "lm_head.weight"
        )
        runtime_operations: list[RuntimeOperationTopology] = [
            RuntimeOperationTopology(
                logical_layer=None,
                capability="embedding.lookup.int8-row.v1",
                abi=1,
                routed_component=None,
                component_layer=0,
                tensor_bindings=(("weight", "model.embed_tokens.weight"),),
                input_bindings=(("token_ids", "request.token_ids", TOKEN_BATCH_ABI),),
                output_bindings=(("hidden", "hidden.0", HIDDEN_BATCH_ABI),),
            )
        ]
        for layer in range(layers):
            prefix = f"model.layers.{layer}."
            block_hidden = f"layer.{layer}.after_block"
            expert_input = f"layer.{layer}.expert_input"
            route_indices = f"layer.{layer}.route_indices"
            route_weights = f"layer.{layer}.route_weights"
            residual = f"layer.{layer}.residual"
            shared_output = f"layer.{layer}.shared_output"
            full_attention = (layer + 1) % full_interval == 0
            if full_attention:
                block_bindings = (
                    ("input_norm", prefix + "input_layernorm.weight"),
                    ("query_projection", prefix + "self_attn.q_proj.weight"),
                    ("key_projection", prefix + "self_attn.k_proj.weight"),
                    ("value_projection", prefix + "self_attn.v_proj.weight"),
                    ("output_projection", prefix + "self_attn.o_proj.weight"),
                    ("query_norm", prefix + "self_attn.q_norm.weight"),
                    ("key_norm", prefix + "self_attn.k_norm.weight"),
                )
            else:
                block_bindings = (
                    ("input_norm", prefix + "input_layernorm.weight"),
                    ("qkvz_projection", prefix + "linear_attn.in_proj_qkvz.weight"),
                    ("ba_projection", prefix + "linear_attn.in_proj_ba.weight"),
                    ("convolution", prefix + "linear_attn.conv1d.weight"),
                    ("time_bias", prefix + "linear_attn.dt_bias"),
                    ("decay_log", prefix + "linear_attn.A_log"),
                    ("output_norm", prefix + "linear_attn.norm.weight"),
                    ("output_projection", prefix + "linear_attn.out_proj.weight"),
                )
            runtime_operations.extend((
                RuntimeOperationTopology(
                    logical_layer=layer,
                    capability=runtime_layers[layer].block_capability,
                    abi=1,
                    routed_component=None,
                    component_layer=0,
                    parameters=(
                        ("norm_epsilon_f32_bits", norm_bits),
                        ("rope_theta_f32_bits", rope_bits),
                        ("rotary_dimension", rotary_dim),
                    ),
                    tensor_bindings=block_bindings,
                    input_bindings=(
                        ("hidden", f"hidden.{layer}", HIDDEN_BATCH_ABI),
                        ("positions", "request.positions", POSITION_BATCH_ABI),
                    ),
                    output_bindings=(("hidden", block_hidden, HIDDEN_BATCH_ABI),),
                ),
                RuntimeOperationTopology(
                    logical_layer=layer,
                    capability="router.linear-topk.shared-swiglu.v1",
                    abi=1,
                    routed_component="decoder",
                    component_layer=layer,
                    parameters=(
                        ("norm_epsilon_f32_bits", norm_bits),
                        ("shared_intermediate_size", shared_width),
                    ),
                    tensor_bindings=(
                        ("input_norm", prefix + "post_attention_layernorm.weight"),
                        ("router_weight", prefix + "mlp.gate.weight"),
                        ("shared_gate_projection", prefix + "mlp.shared_expert.gate_proj.weight"),
                        ("shared_up_projection", prefix + "mlp.shared_expert.up_proj.weight"),
                        ("shared_down_projection", prefix + "mlp.shared_expert.down_proj.weight"),
                        ("shared_router", prefix + "mlp.shared_expert_gate.weight"),
                    ),
                    input_bindings=(("hidden", block_hidden, HIDDEN_BATCH_ABI),),
                    output_bindings=(
                        ("expert_input", expert_input, HIDDEN_BATCH_ABI),
                        ("route_indices", route_indices, ROUTE_INDEX_BATCH_ABI),
                        ("route_weights", route_weights, ROUTE_WEIGHT_BATCH_ABI),
                        ("residual", residual, HIDDEN_BATCH_ABI),
                        ("shared_output", shared_output, HIDDEN_BATCH_ABI),
                    ),
                ),
                RuntimeOperationTopology(
                    logical_layer=layer,
                    capability="moe.swiglu.routed.merge-shared.v1",
                    abi=1,
                    routed_component="decoder",
                    component_layer=layer,
                    input_bindings=(
                        ("expert_input", expert_input, HIDDEN_BATCH_ABI),
                        ("route_indices", route_indices, ROUTE_INDEX_BATCH_ABI),
                        ("route_weights", route_weights, ROUTE_WEIGHT_BATCH_ABI),
                        ("residual", residual, HIDDEN_BATCH_ABI),
                        ("shared_output", shared_output, HIDDEN_BATCH_ABI),
                    ),
                    output_bindings=(
                        ("hidden", f"hidden.{layer + 1}", HIDDEN_BATCH_ABI),
                    ),
                ),
            ))
        runtime_operations.append(RuntimeOperationTopology(
            logical_layer=None,
            capability="head.rmsnorm.argmax.int8-row.v1",
            abi=1,
            routed_component=None,
            component_layer=0,
            tensor_bindings=(
                ("norm", "model.norm.weight"),
                ("weight", output_head),
            ),
            input_bindings=(("hidden", f"hidden.{layers}", HIDDEN_BATCH_ABI),),
            output_bindings=(("token_ids", "response.token_ids", TOKEN_BATCH_ABI),),
        ))
        runtime_topology = RuntimeModelTopology(
            architecture_id=self.name,
            vocab_size=vocab,
            max_context_tokens=max_context,
            hidden_size=hidden,
            attributes=(
                ("attention_heads", heads),
                ("kv_heads", kv_heads),
                ("head_dim", head_dim),
                ("full_attention_interval", full_interval),
                ("linear_conv_kernel", conv_kernel),
                ("linear_key_head_dim", key_head_dim),
                ("linear_value_head_dim", value_head_dim),
                ("linear_key_heads", key_heads),
                ("linear_value_heads", value_heads),
                ("shared_intermediate_size", shared_width),
                ("norm_epsilon_f32_bits", norm_bits),
                ("rope_theta_f32_bits", rope_bits),
                ("rotary_dimension", rotary_dim),
            ),
            required_kernels=(
                ("embedding.lookup.int8-row.v1", 1),
                ("block.full-attention.output-gated.v1", 1),
                ("block.recurrent-linear-attention.gated-delta.v1", 1),
                ("router.linear-topk.shared-swiglu.v1", 1),
                ("moe.swiglu.routed.merge-shared.v1", 1),
                ("head.rmsnorm.argmax.int8-row.v1", 1),
            ),
            components=(runtime_component,),
            layers=runtime_layers,
            operations=tuple(runtime_operations),
            tensor_bindings=(
                ("token_embedding", "model.embed_tokens.weight"),
                ("final_norm", "model.norm.weight"),
                ("output_head", output_head),
            ),
            program_inputs=(
                ("token_ids", "request.token_ids", TOKEN_BATCH_ABI),
                ("positions", "request.positions", POSITION_BATCH_ABI),
            ),
            program_outputs=(
                ("next_token_ids", "response.token_ids", TOKEN_BATCH_ABI),
            ),
        )
        return AdaptedModel(
            family=self.name,
            architecture=architecture,
            dense=tuple(dense_by_name[name] for name in sorted(dense_by_name)),
            experts=tuple(sources),
            source_tensor_count=len(checkpoint.tensors),
            runtime_topology=runtime_topology,
            supported_expert_quant_profiles=frozenset((FP4_QUANT_PROFILE,)),
        )


class Qwen3_5Adapter:
    """Strict adapter for the official dense Qwen3.5/Qwen3.8 checkpoint ABI."""

    name = "qwen3_5"

    def adapt(self, checkpoint: SafeTensorCheckpoint) -> AdaptedModel:
        config = checkpoint.config
        if config.get("model_type") != self.name:
            raise AdapterError(
                "qwen3_5 adapter requires config.model_type == 'qwen3_5'"
            )
        architectures = config.get("architectures")
        if not isinstance(architectures, list) or (
            "Qwen3_5ForConditionalGeneration" not in architectures
        ):
            raise AdapterError(
                "qwen3_5 adapter requires Qwen3_5ForConditionalGeneration"
            )
        text = config.get("text_config")
        vision = config.get("vision_config")
        if not isinstance(text, dict) or not isinstance(vision, dict):
            raise AdapterError("qwen3_5 requires text_config and vision_config")
        if text.get("model_type") != "qwen3_5_text":
            raise AdapterError("qwen3_5 text_config has an unexpected model_type")

        layers = _integer(text, "num_hidden_layers")
        hidden = _integer(text, "hidden_size")
        intermediate = _integer(text, "intermediate_size")
        vocab = _integer(text, "vocab_size")
        max_context = _integer(text, "max_position_embeddings")
        heads = _integer(text, "num_attention_heads")
        kv_heads = _integer(text, "num_key_value_heads")
        head_dim = _integer(text, "head_dim")
        conv_kernel = _integer(text, "linear_conv_kernel_dim")
        key_head_dim = _integer(text, "linear_key_head_dim")
        value_head_dim = _integer(text, "linear_value_head_dim")
        key_heads = _integer(text, "linear_num_key_heads")
        value_heads = _integer(text, "linear_num_value_heads")
        full_interval = _integer(text, "full_attention_interval")
        mtp_layers = _integer(text, "mtp_num_hidden_layers")
        epsilon = _number(text, "rms_norm_eps")
        # Qwen3.5 declares head_dim explicitly.  Its output-gated attention
        # may project to a query width larger than hidden_size (the official
        # 27B checkpoint uses 24 * 256 versus hidden_size 5120), so residual
        # width divisibility by the query-head count is not an invariant.
        if value_heads % key_heads or heads % kv_heads:
            raise AdapterError("qwen3_5 attention geometry is inconsistent")
        if hidden // heads != head_dim:
            # Qwen3.5 has an output-gated query width which need not equal the
            # residual width, so only the explicit head_dim is authoritative.
            if heads * head_dim <= hidden:
                raise AdapterError("qwen3_5 explicit head geometry is invalid")
        layer_types = text.get("layer_types")
        if (
            not isinstance(layer_types, list)
            or len(layer_types) != layers
            or any(
                value not in {"linear_attention", "full_attention"}
                for value in layer_types
            )
            or any(
                (value == "full_attention") != ((layer + 1) % full_interval == 0)
                for layer, value in enumerate(layer_types)
            )
        ):
            raise AdapterError("qwen3_5 layer_types disagree with configured topology")
        partial_rotary = text.get("partial_rotary_factor")
        if (
            isinstance(partial_rotary, bool)
            or not isinstance(partial_rotary, (int, float))
            or not 0.0 < float(partial_rotary) <= 1.0
        ):
            raise AdapterError("qwen3_5 partial_rotary_factor must be in (0,1]")
        rotary_dim = int(head_dim * float(partial_rotary))
        if rotary_dim <= 0 or rotary_dim % 2:
            raise AdapterError("qwen3_5 rotary dimension is invalid")
        rope = text.get("rope_parameters")
        if not isinstance(rope, dict):
            raise AdapterError("qwen3_5 rope_parameters are absent")
        rope_theta = _number(rope, "rope_theta")
        mrope_sections = rope.get("mrope_section")
        if (
            not isinstance(mrope_sections, list)
            or len(mrope_sections) != 3
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in mrope_sections)
            or sum(mrope_sections) != rotary_dim // 2
        ):
            raise AdapterError("qwen3_5 mRoPE sections are inconsistent")

        vision_depth = _integer(vision, "depth")
        vision_hidden = _integer(vision, "hidden_size")
        vision_intermediate = _integer(vision, "intermediate_size")
        vision_heads = _integer(vision, "num_heads")
        vision_positions = _integer(vision, "num_position_embeddings")
        vision_output = _integer(vision, "out_hidden_size")
        vision_channels = _integer(vision, "in_channels")
        patch = _integer(vision, "patch_size")
        temporal_patch = _integer(vision, "temporal_patch_size")
        spatial_merge = _integer(vision, "spatial_merge_size")
        vision_grid_side = math.isqrt(vision_positions)
        if (
            vision_hidden % vision_heads
            or vision_output != hidden
            or vision_grid_side * vision_grid_side != vision_positions
        ):
            raise AdapterError("qwen3_5 vision/text geometry is inconsistent")

        expected: dict[str, tuple[int, ...]] = {}
        fp4_names: set[str] = set()
        float32_names: set[str] = set()

        def add(
            name: str, shape: tuple[int, ...], *, fp4: bool = False,
            preserve: bool = False,
        ) -> None:
            if name in expected or (fp4 and preserve):
                raise AdapterError(f"duplicate/ambiguous qwen3_5 tensor role: {name}")
            expected[name] = shape
            if fp4:
                fp4_names.add(name)
            if preserve:
                float32_names.add(name)

        text_prefix = "model.language_model."
        add(text_prefix + "embed_tokens.weight", (vocab, hidden), fp4=True)
        add(text_prefix + "norm.weight", (hidden,), preserve=True)
        tied = bool(text.get("tie_word_embeddings", False))
        if not tied:
            add("lm_head.weight", (vocab, hidden), fp4=True)
        key_dim = key_heads * key_head_dim
        value_dim = value_heads * value_head_dim
        conv_dim = 2 * key_dim + value_dim
        for layer, layer_type in enumerate(layer_types):
            prefix = f"{text_prefix}layers.{layer}."
            add(prefix + "input_layernorm.weight", (hidden,), preserve=True)
            add(prefix + "post_attention_layernorm.weight", (hidden,), preserve=True)
            add(prefix + "mlp.gate_proj.weight", (intermediate, hidden), fp4=True)
            add(prefix + "mlp.up_proj.weight", (intermediate, hidden), fp4=True)
            add(prefix + "mlp.down_proj.weight", (hidden, intermediate), fp4=True)
            if layer_type == "full_attention":
                add(prefix + "self_attn.q_proj.weight",
                    (2 * heads * head_dim, hidden), fp4=True)
                add(prefix + "self_attn.k_proj.weight",
                    (kv_heads * head_dim, hidden), fp4=True)
                add(prefix + "self_attn.v_proj.weight",
                    (kv_heads * head_dim, hidden), fp4=True)
                add(prefix + "self_attn.o_proj.weight",
                    (hidden, heads * head_dim), fp4=True)
                add(prefix + "self_attn.q_norm.weight", (head_dim,), preserve=True)
                add(prefix + "self_attn.k_norm.weight", (head_dim,), preserve=True)
            else:
                add(prefix + "linear_attn.in_proj_qkv.weight",
                    (2 * key_dim + value_dim, hidden), fp4=True)
                add(prefix + "linear_attn.in_proj_z.weight",
                    (value_dim, hidden), fp4=True)
                add(prefix + "linear_attn.in_proj_b.weight",
                    (value_heads, hidden), fp4=True)
                add(prefix + "linear_attn.in_proj_a.weight",
                    (value_heads, hidden), fp4=True)
                add(prefix + "linear_attn.conv1d.weight",
                    (conv_dim, 1, conv_kernel), fp4=True)
                add(prefix + "linear_attn.dt_bias", (value_heads,), preserve=True)
                add(prefix + "linear_attn.A_log", (value_heads,), preserve=True)
                add(prefix + "linear_attn.norm.weight",
                    (value_head_dim,), preserve=True)
                add(prefix + "linear_attn.out_proj.weight",
                    (hidden, value_dim), fp4=True)

        if mtp_layers and bool(text.get("mtp_use_dedicated_embeddings", False)):
            raise AdapterError("qwen3_5 dedicated MTP embeddings are unsupported")
        if mtp_layers:
            add("mtp.fc.weight", (hidden, 2 * hidden), fp4=True)
            add("mtp.pre_fc_norm_embedding.weight", (hidden,), preserve=True)
            add("mtp.pre_fc_norm_hidden.weight", (hidden,), preserve=True)
            add("mtp.norm.weight", (hidden,), preserve=True)
            for layer in range(mtp_layers):
                prefix = f"mtp.layers.{layer}."
                add(prefix + "input_layernorm.weight", (hidden,), preserve=True)
                add(prefix + "post_attention_layernorm.weight", (hidden,), preserve=True)
                add(prefix + "self_attn.q_proj.weight",
                    (2 * heads * head_dim, hidden), fp4=True)
                add(prefix + "self_attn.k_proj.weight",
                    (kv_heads * head_dim, hidden), fp4=True)
                add(prefix + "self_attn.v_proj.weight",
                    (kv_heads * head_dim, hidden), fp4=True)
                add(prefix + "self_attn.o_proj.weight",
                    (hidden, heads * head_dim), fp4=True)
                add(prefix + "self_attn.q_norm.weight", (head_dim,), preserve=True)
                add(prefix + "self_attn.k_norm.weight", (head_dim,), preserve=True)
                add(prefix + "mlp.gate_proj.weight", (intermediate, hidden), fp4=True)
                add(prefix + "mlp.up_proj.weight", (intermediate, hidden), fp4=True)
                add(prefix + "mlp.down_proj.weight", (hidden, intermediate), fp4=True)

        visual_prefix = "model.visual."
        add(visual_prefix + "patch_embed.proj.weight",
            (vision_hidden, vision_channels, temporal_patch, patch, patch), fp4=True)
        add(visual_prefix + "patch_embed.proj.bias",
            (vision_hidden,), preserve=True)
        add(visual_prefix + "pos_embed.weight",
            (vision_positions, vision_hidden), fp4=True)
        for layer in range(vision_depth):
            prefix = f"{visual_prefix}blocks.{layer}."
            add(prefix + "attn.qkv.weight",
                (3 * vision_hidden, vision_hidden), fp4=True)
            add(prefix + "attn.qkv.bias", (3 * vision_hidden,), preserve=True)
            add(prefix + "attn.proj.weight",
                (vision_hidden, vision_hidden), fp4=True)
            add(prefix + "attn.proj.bias", (vision_hidden,), preserve=True)
            add(prefix + "mlp.linear_fc1.weight",
                (vision_intermediate, vision_hidden), fp4=True)
            add(prefix + "mlp.linear_fc1.bias",
                (vision_intermediate,), preserve=True)
            add(prefix + "mlp.linear_fc2.weight",
                (vision_hidden, vision_intermediate), fp4=True)
            add(prefix + "mlp.linear_fc2.bias", (vision_hidden,), preserve=True)
            for norm in ("norm1", "norm2"):
                add(prefix + f"{norm}.weight", (vision_hidden,), preserve=True)
                add(prefix + f"{norm}.bias", (vision_hidden,), preserve=True)
        merged = vision_hidden * spatial_merge * spatial_merge
        add(visual_prefix + "merger.norm.weight", (vision_hidden,), preserve=True)
        add(visual_prefix + "merger.norm.bias", (vision_hidden,), preserve=True)
        add(visual_prefix + "merger.linear_fc1.weight", (merged, merged), fp4=True)
        add(visual_prefix + "merger.linear_fc1.bias", (merged,), preserve=True)
        add(visual_prefix + "merger.linear_fc2.weight",
            (vision_output, merged), fp4=True)
        add(visual_prefix + "merger.linear_fc2.bias", (vision_output,), preserve=True)

        actual = set(checkpoint.tensors)
        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            unknown = sorted(actual - set(expected))
            raise AdapterError(
                f"qwen3_5 tensor partition mismatch; missing={missing[:8]}, "
                f"unknown={unknown[:8]}"
            )
        for name, shape in expected.items():
            info = checkpoint.tensors[name]
            if info.shape != shape:
                raise AdapterError(
                    f"qwen3_5 shape mismatch for {name}: expected {shape}, "
                    f"got {info.shape}"
                )
            if info.dtype not in {"BF16", "F16", "F32"}:
                raise AdapterError(
                    f"qwen3_5 tensor {name} has unsupported dtype {info.dtype}"
                )

        norm_bits = _float32_bits(epsilon)
        rope_bits = _float32_bits(rope_theta)
        full_layers = sum(value == "full_attention" for value in layer_types)
        architecture = {
            "family": self.name,
            "model_type": self.name,
            "architectures": architectures,
            "language_model_only": bool(config.get("language_model_only", False)),
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "vocab_size": vocab,
            "max_position_embeddings": max_context,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "hidden_activation": text.get("hidden_act", "silu"),
            "rms_norm_epsilon": epsilon,
            "rope": {"theta": rope_theta, "parameters": rope},
            "partial_rotary_factor": float(partial_rotary),
            "full_attention_interval": full_interval,
            "full_attention_layers": full_layers,
            "linear_attention_layers": layers - full_layers,
            "linear_conv_kernel_dim": conv_kernel,
            "linear_key_head_dim": key_head_dim,
            "linear_value_head_dim": value_head_dim,
            "linear_num_key_heads": key_heads,
            "linear_num_value_heads": value_heads,
            "multi_token_prediction_layers": mtp_layers,
            "attention_output_gate": bool(text.get("attn_output_gate", False)),
            "tie_word_embeddings": tied,
            "vision_depth": vision_depth,
            "vision_hidden_size": vision_hidden,
            "vision_intermediate_size": vision_intermediate,
        }
        runtime_layers = tuple(
            RuntimeLayerTopology(
                logical_layer=layer,
                block_capability=(
                    "block.full-attention.output-gated.v1"
                    if layer_type == "full_attention" else
                    "block.recurrent-linear-attention.split-gated-delta.v1"
                ),
                block_abi=1,
                routed_component=None,
                component_layer=0,
            )
            for layer, layer_type in enumerate(layer_types)
        )
        output_head = (
            text_prefix + "embed_tokens.weight" if tied else "lm_head.weight"
        )
        operations: list[RuntimeOperationTopology] = [
            RuntimeOperationTopology(
                logical_layer=None,
                capability="embedding.lookup.fp4-block32.v1",
                abi=1,
                routed_component=None,
                component_layer=0,
                tensor_bindings=(("weight", text_prefix + "embed_tokens.weight"),),
                input_bindings=(("token_ids", "request.token_ids", TOKEN_BATCH_ABI),),
                output_bindings=(("hidden", "hidden.text", HIDDEN_BATCH_ABI),),
            )
        ]
        vision_bindings: list[tuple[str, str]] = [
            ("patch_projection", visual_prefix + "patch_embed.proj.weight"),
            ("patch_bias", visual_prefix + "patch_embed.proj.bias"),
            ("position_embedding", visual_prefix + "pos_embed.weight"),
        ]
        for layer in range(vision_depth):
            prefix = f"{visual_prefix}blocks.{layer}."
            role = f"block.{layer}."
            vision_bindings.extend((
                (role + "norm1_weight", prefix + "norm1.weight"),
                (role + "norm1_bias", prefix + "norm1.bias"),
                (role + "qkv_projection", prefix + "attn.qkv.weight"),
                (role + "qkv_bias", prefix + "attn.qkv.bias"),
                (role + "attention_projection", prefix + "attn.proj.weight"),
                (role + "attention_bias", prefix + "attn.proj.bias"),
                (role + "norm2_weight", prefix + "norm2.weight"),
                (role + "norm2_bias", prefix + "norm2.bias"),
                (role + "mlp_fc1", prefix + "mlp.linear_fc1.weight"),
                (role + "mlp_fc1_bias", prefix + "mlp.linear_fc1.bias"),
                (role + "mlp_fc2", prefix + "mlp.linear_fc2.weight"),
                (role + "mlp_fc2_bias", prefix + "mlp.linear_fc2.bias"),
            ))
        vision_bindings.extend((
            ("merger_norm_weight", visual_prefix + "merger.norm.weight"),
            ("merger_norm_bias", visual_prefix + "merger.norm.bias"),
            ("merger_fc1", visual_prefix + "merger.linear_fc1.weight"),
            ("merger_fc1_bias", visual_prefix + "merger.linear_fc1.bias"),
            ("merger_fc2", visual_prefix + "merger.linear_fc2.weight"),
            ("merger_fc2_bias", visual_prefix + "merger.linear_fc2.bias"),
        ))
        operations.append(RuntimeOperationTopology(
            logical_layer=None,
            capability="vision.patch-transformer-merge.fp4-block32.v1",
            abi=1,
            routed_component=None,
            component_layer=0,
            tensor_bindings=tuple(vision_bindings),
            input_bindings=(
                ("hidden", "hidden.text", HIDDEN_BATCH_ABI),
                ("media", "request.multimodal", MULTIMODAL_BATCH_ABI),
            ),
            output_bindings=(("hidden", "hidden.0", HIDDEN_BATCH_ABI),),
        ))
        for layer, layer_type in enumerate(layer_types):
            prefix = f"{text_prefix}layers.{layer}."
            after_attention = f"layer.{layer}.after_attention"
            if layer_type == "full_attention":
                attention_bindings = (
                    ("input_norm", prefix + "input_layernorm.weight"),
                    ("query_projection", prefix + "self_attn.q_proj.weight"),
                    ("key_projection", prefix + "self_attn.k_proj.weight"),
                    ("value_projection", prefix + "self_attn.v_proj.weight"),
                    ("output_projection", prefix + "self_attn.o_proj.weight"),
                    ("query_norm", prefix + "self_attn.q_norm.weight"),
                    ("key_norm", prefix + "self_attn.k_norm.weight"),
                )
            else:
                attention_bindings = (
                    ("input_norm", prefix + "input_layernorm.weight"),
                    ("qkv_projection", prefix + "linear_attn.in_proj_qkv.weight"),
                    ("z_projection", prefix + "linear_attn.in_proj_z.weight"),
                    ("b_projection", prefix + "linear_attn.in_proj_b.weight"),
                    ("a_projection", prefix + "linear_attn.in_proj_a.weight"),
                    ("convolution", prefix + "linear_attn.conv1d.weight"),
                    ("time_bias", prefix + "linear_attn.dt_bias"),
                    ("decay_log", prefix + "linear_attn.A_log"),
                    ("output_norm", prefix + "linear_attn.norm.weight"),
                    ("output_projection", prefix + "linear_attn.out_proj.weight"),
                )
            operations.extend((
                RuntimeOperationTopology(
                    logical_layer=layer,
                    capability=runtime_layers[layer].block_capability,
                    abi=1,
                    routed_component=None,
                    component_layer=0,
                    tensor_bindings=attention_bindings,
                    input_bindings=(
                        ("hidden", f"hidden.{layer}", HIDDEN_BATCH_ABI),
                        ("positions", "request.positions", POSITION_BATCH_ABI),
                    ),
                    output_bindings=(("hidden", after_attention, HIDDEN_BATCH_ABI),),
                ),
                RuntimeOperationTopology(
                    logical_layer=layer,
                    capability="ffn.swiglu.dense.fp4-block32.v1",
                    abi=1,
                    routed_component=None,
                    component_layer=0,
                    tensor_bindings=(
                        ("input_norm", prefix + "post_attention_layernorm.weight"),
                        ("gate_projection", prefix + "mlp.gate_proj.weight"),
                        ("up_projection", prefix + "mlp.up_proj.weight"),
                        ("down_projection", prefix + "mlp.down_proj.weight"),
                    ),
                    input_bindings=(("hidden", after_attention, HIDDEN_BATCH_ABI),),
                    output_bindings=(
                        ("hidden", f"hidden.{layer + 1}", HIDDEN_BATCH_ABI),
                    ),
                ),
            ))
        operations.append(RuntimeOperationTopology(
            logical_layer=None,
            capability="head.rmsnorm.token-select.fp4-block32.v1",
            abi=1,
            routed_component=None,
            component_layer=0,
            tensor_bindings=(
                ("norm", text_prefix + "norm.weight"),
                ("weight", output_head),
            ),
            input_bindings=(("hidden", f"hidden.{layers}", HIDDEN_BATCH_ABI),),
            output_bindings=(("token_ids", "response.token_ids", TOKEN_BATCH_ABI),),
        ))
        mtp_capability = "decode.mtp.dense-full-attention.fp4-block32.exact.v1"
        exact_decode = None
        if mtp_layers:
            mtp_bindings: list[tuple[str, str]] = [
                ("token_embedding", text_prefix + "embed_tokens.weight"),
                ("output_head", output_head),
                ("fusion_projection", "mtp.fc.weight"),
                ("embedding_norm", "mtp.pre_fc_norm_embedding.weight"),
                ("hidden_norm", "mtp.pre_fc_norm_hidden.weight"),
                ("draft_norm", "mtp.norm.weight"),
            ]
            for layer in range(mtp_layers):
                prefix = f"mtp.layers.{layer}."
                role = f"layer.{layer}."
                mtp_bindings.extend((
                    (role + "input_norm", prefix + "input_layernorm.weight"),
                    (role + "query_projection", prefix + "self_attn.q_proj.weight"),
                    (role + "key_projection", prefix + "self_attn.k_proj.weight"),
                    (role + "value_projection", prefix + "self_attn.v_proj.weight"),
                    (role + "output_projection", prefix + "self_attn.o_proj.weight"),
                    (role + "query_norm", prefix + "self_attn.q_norm.weight"),
                    (role + "key_norm", prefix + "self_attn.k_norm.weight"),
                    (role + "post_attention_norm",
                     prefix + "post_attention_layernorm.weight"),
                    (role + "gate_projection", prefix + "mlp.gate_proj.weight"),
                    (role + "up_projection", prefix + "mlp.up_proj.weight"),
                    (role + "down_projection", prefix + "mlp.down_proj.weight"),
                ))
            exact_decode = RuntimeExactDecodeTopology(
                capability=mtp_capability,
                abi=1,
                maximum_emitted_tokens=mtp_layers + 1,
                parameters=(("draft_layers", mtp_layers),
                            ("embedding_first", 1), ("post_norm", 1)),
                tensor_bindings=tuple(mtp_bindings),
            )
        required_kernels = [
            ("embedding.lookup.fp4-block32.v1", 1),
            ("vision.patch-transformer-merge.fp4-block32.v1", 1),
            ("block.full-attention.output-gated.v1", 1),
            ("block.recurrent-linear-attention.split-gated-delta.v1", 1),
            ("ffn.swiglu.dense.fp4-block32.v1", 1),
            ("head.rmsnorm.token-select.fp4-block32.v1", 1),
        ]
        if exact_decode is not None:
            required_kernels.append((mtp_capability, 1))
        runtime_topology = RuntimeModelTopology(
            architecture_id=self.name,
            vocab_size=vocab,
            max_context_tokens=max_context,
            hidden_size=hidden,
            attributes=(
                ("attention_heads", heads),
                ("kv_heads", kv_heads),
                ("head_dim", head_dim),
                ("full_attention_layers", full_layers),
                ("linear_conv_kernel", conv_kernel),
                ("linear_key_head_dim", key_head_dim),
                ("linear_value_head_dim", value_head_dim),
                ("linear_key_heads", key_heads),
                ("linear_value_heads", value_heads),
                ("norm_epsilon_f32_bits", norm_bits),
                ("rope_theta_f32_bits", rope_bits),
                ("rotary_dimension", rotary_dim),
                ("mrope_interleaved", int(bool(rope.get("mrope_interleaved", False)))),
                ("mrope_section_0", mrope_sections[0]),
                ("mrope_section_1", mrope_sections[1]),
                ("mrope_section_2", mrope_sections[2]),
                ("vision_depth", vision_depth),
                ("vision_hidden_size", vision_hidden),
                ("vision_intermediate_size", vision_intermediate),
                ("vision_heads", vision_heads),
                ("vision_position_embeddings", vision_positions),
                ("vision_grid_side", vision_grid_side),
                ("vision_channels", vision_channels),
                ("vision_patch_size", patch),
                ("vision_temporal_patch_size", temporal_patch),
                ("vision_spatial_merge_size", spatial_merge),
                ("vision_output_size", vision_output),
                ("vision_norm_epsilon_f32_bits", _float32_bits(1e-6)),
                ("vision_rope_theta_f32_bits", _float32_bits(10_000.0)),
                ("zero_centered_norm", 1),
                ("mtp_layers", mtp_layers),
            ),
            required_kernels=tuple(required_kernels),
            components=(),
            layers=runtime_layers,
            operations=tuple(operations),
            tensor_bindings=(
                ("token_embedding", text_prefix + "embed_tokens.weight"),
                ("final_norm", text_prefix + "norm.weight"),
                ("output_head", output_head),
            ),
            program_inputs=(
                ("token_ids", "request.token_ids", TOKEN_BATCH_ABI),
                ("positions", "request.positions", POSITION_BATCH_ABI),
                ("multimodal", "request.multimodal", MULTIMODAL_BATCH_ABI),
            ),
            program_outputs=(
                ("next_token_ids", "response.token_ids", TOKEN_BATCH_ABI),
            ),
            exact_decode=exact_decode,
        )
        return AdaptedModel(
            family=self.name,
            architecture=architecture,
            dense=tuple(checkpoint.tensors[name] for name in sorted(expected)),
            experts=(),
            source_tensor_count=len(checkpoint.tensors),
            runtime_topology=runtime_topology,
            dense_float32=frozenset(float32_names),
            dense_fp4=frozenset(fp4_names),
            supported_expert_quant_profiles=frozenset((FP4_QUANT_PROFILE,)),
        )


class Lfm2MoeAdapter:
    """Strict adapter for the upstream Transformers LFM2-MoE checkpoint ABI."""

    name = "lfm2_moe"

    def adapt(self, checkpoint: SafeTensorCheckpoint) -> AdaptedModel:
        config = checkpoint.config
        if config.get("model_type") != self.name:
            raise AdapterError(
                "lfm2_moe adapter requires config.model_type == 'lfm2_moe'"
            )
        architectures = config.get("architectures")
        if architectures is not None and (
            not isinstance(architectures, list)
            or "Lfm2MoeForCausalLM" not in architectures
        ):
            raise AdapterError(
                "lfm2_moe adapter requires Lfm2MoeForCausalLM architecture"
            )

        layers = _integer(config, "num_hidden_layers")
        experts = _integer(config, "num_experts")
        top_k = _integer(config, "num_experts_per_tok")
        hidden = _integer(config, "hidden_size")
        dense_width = _integer(config, "intermediate_size")
        expert_width = _integer(config, "moe_intermediate_size")
        vocab = _integer(config, "vocab_size")
        max_context = _integer(config, "max_position_embeddings")
        heads = _integer(config, "num_attention_heads")
        kv_heads = _integer(config, "num_key_value_heads")
        conv_cache = _integer(config, "conv_L_cache")
        dense_layers_value = config.get("num_dense_layers")
        if (
            isinstance(dense_layers_value, bool)
            or not isinstance(dense_layers_value, int)
            or dense_layers_value < 0
            or dense_layers_value >= layers
        ):
            raise AdapterError(
                "num_dense_layers must leave at least one routed layer"
            )
        dense_layers = dense_layers_value
        if hidden % heads or heads % kv_heads:
            raise AdapterError("invalid LFM2 attention head geometry")
        if top_k > experts:
            raise AdapterError("num_experts_per_tok exceeds num_experts")
        head_dim = hidden // heads
        layer_types = config.get("layer_types")
        if (
            not isinstance(layer_types, list)
            or len(layer_types) != layers
            or any(kind not in {"conv", "full_attention"} for kind in layer_types)
        ):
            raise AdapterError(
                "layer_types must classify every layer as conv/full_attention"
            )
        conv_bias = config.get("conv_bias", False)
        norm_topk = config.get("norm_topk_prob", True)
        use_expert_bias = config.get("use_expert_bias", True)
        tie_embeddings = config.get("tie_word_embeddings", True)
        for key, value in (
            ("conv_bias", conv_bias),
            ("norm_topk_prob", norm_topk),
            ("use_expert_bias", use_expert_bias),
            ("tie_word_embeddings", tie_embeddings),
        ):
            if not isinstance(value, bool):
                raise AdapterError(f"{key} must be boolean")
        norm_epsilon = _number(config, "norm_eps")
        rope_theta = _number(config, "rope_theta")
        routed_scaling = _number(config, "routed_scaling_factor")

        expected_dense: dict[str, tuple[int, ...]] = {
            "model.embed_tokens.weight": (vocab, hidden),
            "model.embedding_norm.weight": (hidden,),
        }
        if not tie_embeddings:
            expected_dense["lm_head.weight"] = (vocab, hidden)
        for layer, layer_type in enumerate(layer_types):
            prefix = f"model.layers.{layer}."
            expected_dense.update({
                prefix + "operator_norm.weight": (hidden,),
                prefix + "ffn_norm.weight": (hidden,),
            })
            if layer_type == "conv":
                expected_dense.update({
                    prefix + "conv.in_proj.weight": (3 * hidden, hidden),
                    prefix + "conv.conv.weight": (hidden, 1, conv_cache),
                    prefix + "conv.out_proj.weight": (hidden, hidden),
                })
                if conv_bias:
                    expected_dense.update({
                        prefix + "conv.in_proj.bias": (3 * hidden,),
                        prefix + "conv.conv.bias": (hidden,),
                        prefix + "conv.out_proj.bias": (hidden,),
                    })
            else:
                expected_dense.update({
                    prefix + "self_attn.q_proj.weight": (heads * head_dim, hidden),
                    prefix + "self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
                    prefix + "self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
                    prefix + "self_attn.out_proj.weight": (hidden, heads * head_dim),
                    prefix + "self_attn.q_layernorm.weight": (head_dim,),
                    prefix + "self_attn.k_layernorm.weight": (head_dim,),
                })
            if layer < dense_layers:
                expected_dense.update({
                    prefix + "feed_forward.w1.weight": (dense_width, hidden),
                    prefix + "feed_forward.w2.weight": (hidden, dense_width),
                    prefix + "feed_forward.w3.weight": (dense_width, hidden),
                })
            else:
                expected_dense[prefix + "feed_forward.gate.weight"] = (
                    experts, hidden
                )
                if use_expert_bias:
                    expected_dense[prefix + "feed_forward.expert_bias"] = (
                        experts,
                    )

        routed: dict[tuple[int, int], dict[str, TensorInfo]] = {}
        dense_by_name: dict[str, TensorInfo] = {}
        unknown: list[str] = []
        for name, info in checkpoint.tensors.items():
            match = LFM2_EXPERT_PATTERN.fullmatch(name)
            if match:
                logical_layer = int(match["layer"])
                expert = int(match["expert"])
                if logical_layer < dense_layers or logical_layer >= layers:
                    raise AdapterError(
                        f"expert tensor outside routed layer bounds: {name}"
                    )
                if expert >= experts:
                    raise AdapterError(
                        f"expert tensor outside configured expert bounds: {name}"
                    )
                key = (logical_layer - dense_layers, expert)
                routed.setdefault(key, {})[match["projection"]] = info
            elif name in expected_dense:
                dense_by_name[name] = info
            else:
                unknown.append(name)
        if unknown:
            raise AdapterError(f"unidentified LFM2-MoE tensors: {sorted(unknown)[:8]}")
        missing = sorted(set(expected_dense) - set(dense_by_name))
        if missing:
            raise AdapterError(f"missing LFM2-MoE dense tensors: {missing[:8]}")
        for name, expected_shape in expected_dense.items():
            actual_shape = dense_by_name[name].shape
            if actual_shape != expected_shape:
                raise AdapterError(
                    f"dense shape mismatch for {name}: expected "
                    f"{expected_shape}, got {actual_shape}"
                )

        expert_sources: list[ExpertSource] = []
        routed_layers = layers - dense_layers
        for component_layer in range(routed_layers):
            for expert in range(experts):
                projections = routed.get((component_layer, expert), {})
                if set(projections) != {"w1", "w2", "w3"}:
                    raise AdapterError(
                        f"expert ({component_layer}, {expert}) projections are "
                        f"{sorted(projections)}, expected w1/w2/w3"
                    )
                gate = projections["w1"]
                up = projections["w3"]
                down = projections["w2"]
                for projection, info, shape in (
                    ("w1", gate, (expert_width, hidden)),
                    ("w3", up, (expert_width, hidden)),
                    ("w2", down, (hidden, expert_width)),
                ):
                    if info.shape != shape:
                        raise AdapterError(
                            f"expert ({component_layer}, {expert}) {projection} "
                            f"shape mismatch: expected {shape}, got {info.shape}"
                        )
                expert_sources.append(
                    ExpertSource(component_layer, expert, gate, up, down)
                )
        if len(dense_by_name) + 3 * len(expert_sources) != len(checkpoint.tensors):
            raise AdapterError(
                "adapter classification is not a one-to-one tensor partition"
            )

        architecture = {
            "family": self.name,
            "model_type": self.name,
            "architectures": architectures or ["Lfm2MoeForCausalLM"],
            "hidden_size": hidden,
            "intermediate_size": dense_width,
            "moe_intermediate_size": expert_width,
            "vocab_size": vocab,
            "max_position_embeddings": max_context,
            "num_hidden_layers": layers,
            "num_dense_layers": dense_layers,
            "num_routed_layers": routed_layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "num_experts": experts,
            "num_experts_per_token": top_k,
            "shared_experts": 0,
            "layer_types": layer_types,
            "conv_cache_length": conv_cache,
            "conv_bias": conv_bias,
            "hidden_activation": "silu",
            "rms_norm_epsilon": norm_epsilon,
            "rope": {"theta": rope_theta, "scaling": None},
            "normalize_topk_probability": norm_topk,
            "use_expert_bias": use_expert_bias,
            "routed_scaling_factor": routed_scaling,
            "tie_word_embeddings": tie_embeddings,
        }
        norm_bits = _float32_bits(norm_epsilon)
        runtime_component = RuntimeComponentTopology(
            name="decoder",
            layer_count=routed_layers,
            experts_per_layer=experts,
            route_width=top_k,
            shared_experts_per_layer=0,
            hidden_size=hidden,
            intermediate_size=expert_width,
            execution_capability="moe.swiglu.routed.v1",
            router_capability="router.sigmoid-bias.topk.v1",
            router_parameters=(
                ("normalize", int(norm_topk)),
                ("use_expert_bias", int(use_expert_bias)),
                ("normalization_epsilon_f32_bits", _float32_bits(1e-6)),
                ("routed_scaling_factor_f32_bits", _float32_bits(routed_scaling)),
            ),
        )
        runtime_layers: list[RuntimeLayerTopology] = []
        runtime_operations: list[RuntimeOperationTopology] = [
            RuntimeOperationTopology(
                logical_layer=None,
                capability="embedding.lookup.int8-row.v1",
                abi=1,
                routed_component=None,
                component_layer=0,
                tensor_bindings=(("weight", "model.embed_tokens.weight"),),
                input_bindings=(("token_ids", "request.token_ids", TOKEN_BATCH_ABI),),
                output_bindings=(("hidden", "hidden.0", HIDDEN_BATCH_ABI),),
            )
        ]
        for layer, layer_type in enumerate(layer_types):
            prefix = f"model.layers.{layer}."
            block_hidden = f"layer.{layer}.after_block"
            block = (
                "block.full-attention.gqa.qk-norm.v1"
                if layer_type == "full_attention"
                else "block.causal-short-conv.gated.v1"
            )
            parameters = (
                ("norm_epsilon_f32_bits", norm_bits),
                ("head_dim", head_dim),
                ("rope_theta_f32_bits", _float32_bits(rope_theta)),
            ) if layer_type == "full_attention" else (
                ("norm_epsilon_f32_bits", norm_bits),
                ("conv_cache_length", conv_cache),
                ("bias", int(conv_bias)),
            )
            block_bindings = (
                ("input_norm", prefix + "operator_norm.weight"),
                ("query_projection", prefix + "self_attn.q_proj.weight"),
                ("key_projection", prefix + "self_attn.k_proj.weight"),
                ("value_projection", prefix + "self_attn.v_proj.weight"),
                ("output_projection", prefix + "self_attn.out_proj.weight"),
                ("query_norm", prefix + "self_attn.q_layernorm.weight"),
                ("key_norm", prefix + "self_attn.k_layernorm.weight"),
            ) if layer_type == "full_attention" else (
                ("input_norm", prefix + "operator_norm.weight"),
                ("input_projection", prefix + "conv.in_proj.weight"),
                ("convolution", prefix + "conv.conv.weight"),
                ("output_projection", prefix + "conv.out_proj.weight"),
            )
            dense = layer < dense_layers
            component_layer = 0 if dense else layer - dense_layers
            runtime_layers.append(RuntimeLayerTopology(
                logical_layer=layer,
                block_capability=block,
                block_abi=1,
                routed_component=None if dense else "decoder",
                component_layer=component_layer,
                parameters=parameters,
            ))
            runtime_operations.append(RuntimeOperationTopology(
                logical_layer=layer,
                capability=block,
                abi=1,
                routed_component=None,
                component_layer=0,
                parameters=parameters,
                tensor_bindings=block_bindings,
                input_bindings=(
                    ("hidden", f"hidden.{layer}", HIDDEN_BATCH_ABI),
                    ("positions", "request.positions", POSITION_BATCH_ABI),
                ),
                output_bindings=(("hidden", block_hidden, HIDDEN_BATCH_ABI),),
            ))
            if dense:
                runtime_operations.append(RuntimeOperationTopology(
                    logical_layer=layer,
                    capability="ffn.swiglu.dense.v1",
                    abi=1,
                    routed_component=None,
                    component_layer=0,
                    parameters=(
                        ("intermediate_size", dense_width),
                        ("norm_epsilon_f32_bits", norm_bits),
                    ),
                    tensor_bindings=(
                        ("input_norm", prefix + "ffn_norm.weight"),
                        ("gate_projection", prefix + "feed_forward.w1.weight"),
                        ("up_projection", prefix + "feed_forward.w3.weight"),
                        ("down_projection", prefix + "feed_forward.w2.weight"),
                    ),
                    input_bindings=(("hidden", block_hidden, HIDDEN_BATCH_ABI),),
                    output_bindings=(
                        ("hidden", f"hidden.{layer + 1}", HIDDEN_BATCH_ABI),
                    ),
                ))
            else:
                expert_input = f"layer.{layer}.expert_input"
                route_indices = f"layer.{layer}.route_indices"
                route_weights = f"layer.{layer}.route_weights"
                residual = f"layer.{layer}.residual"
                router_bindings = (
                    ("input_norm", prefix + "ffn_norm.weight"),
                    ("router_weight", prefix + "feed_forward.gate.weight"),
                )
                if use_expert_bias:
                    router_bindings += ((
                        "expert_bias",
                        prefix + "feed_forward.expert_bias",
                    ),)
                runtime_operations.extend((
                    RuntimeOperationTopology(
                        logical_layer=layer,
                        capability="router.sigmoid-bias.topk.v1",
                        abi=1,
                        routed_component="decoder",
                        component_layer=component_layer,
                        parameters=(("norm_epsilon_f32_bits", norm_bits),),
                        tensor_bindings=router_bindings,
                        input_bindings=(
                            ("hidden", block_hidden, HIDDEN_BATCH_ABI),
                        ),
                        output_bindings=(
                            ("expert_input", expert_input, HIDDEN_BATCH_ABI),
                            ("route_indices", route_indices, ROUTE_INDEX_BATCH_ABI),
                            ("route_weights", route_weights, ROUTE_WEIGHT_BATCH_ABI),
                            ("residual", residual, HIDDEN_BATCH_ABI),
                        ),
                    ),
                    RuntimeOperationTopology(
                        logical_layer=layer,
                        capability="moe.swiglu.routed.v1",
                        abi=1,
                        routed_component="decoder",
                        component_layer=component_layer,
                        input_bindings=(
                            ("expert_input", expert_input, HIDDEN_BATCH_ABI),
                            ("route_indices", route_indices, ROUTE_INDEX_BATCH_ABI),
                            ("route_weights", route_weights, ROUTE_WEIGHT_BATCH_ABI),
                            ("residual", residual, HIDDEN_BATCH_ABI),
                        ),
                        output_bindings=(
                            ("hidden", f"hidden.{layer + 1}", HIDDEN_BATCH_ABI),
                        ),
                    ),
                ))
        runtime_operations.append(RuntimeOperationTopology(
            logical_layer=None,
            capability="head.rmsnorm.argmax.int8-row.v1",
            abi=1,
            routed_component=None,
            component_layer=0,
            tensor_bindings=(
                ("norm", "model.embedding_norm.weight"),
                ("weight", "model.embed_tokens.weight"),
            ),
            input_bindings=(("hidden", f"hidden.{layers}", HIDDEN_BATCH_ABI),),
            output_bindings=(("token_ids", "response.token_ids", TOKEN_BATCH_ABI),),
        ))
        runtime_topology = RuntimeModelTopology(
            architecture_id=self.name,
            vocab_size=vocab,
            max_context_tokens=max_context,
            hidden_size=hidden,
            attributes=(
                ("attention_heads", heads),
                ("kv_heads", kv_heads),
                ("head_dim", head_dim),
                ("dense_prefix_layers", dense_layers),
                ("norm_epsilon_f32_bits", norm_bits),
            ),
            required_kernels=(
                ("embedding.lookup.int8-row.v1", 1),
                ("block.causal-short-conv.gated.v1", 1),
                ("block.full-attention.gqa.qk-norm.v1", 1),
                ("ffn.swiglu.dense.v1", 1),
                ("router.sigmoid-bias.topk.v1", 1),
                ("moe.swiglu.routed.v1", 1),
                ("head.rmsnorm.argmax.int8-row.v1", 1),
            ),
            components=(runtime_component,),
            layers=tuple(runtime_layers),
            operations=tuple(runtime_operations),
            tensor_bindings=(
                ("token_embedding", "model.embed_tokens.weight"),
                ("final_norm", "model.embedding_norm.weight"),
                ("output_head", "model.embed_tokens.weight"),
            ),
            program_inputs=(
                ("token_ids", "request.token_ids", TOKEN_BATCH_ABI),
                ("positions", "request.positions", POSITION_BATCH_ABI),
            ),
            program_outputs=(
                ("next_token_ids", "response.token_ids", TOKEN_BATCH_ABI),
            ),
        )
        float32_tensors = frozenset(
            f"model.layers.{layer}.feed_forward.gate.weight"
            for layer in range(dense_layers, layers)
        )
        return AdaptedModel(
            family=self.name,
            architecture=architecture,
            dense=tuple(dense_by_name[name] for name in sorted(dense_by_name)),
            experts=tuple(expert_sources),
            source_tensor_count=len(checkpoint.tensors),
            runtime_topology=runtime_topology,
            dense_float32=float32_tensors,
        )


ADAPTERS = {
    OlmoeAdapter.name: OlmoeAdapter(),
    Qwen3NextAdapter.name: Qwen3NextAdapter(),
    Qwen3_5Adapter.name: Qwen3_5Adapter(),
    Lfm2MoeAdapter.name: Lfm2MoeAdapter(),
}


def adapt_checkpoint(checkpoint: SafeTensorCheckpoint, adapter: str) -> AdaptedModel:
    try:
        implementation = ADAPTERS[adapter]
    except KeyError as error:
        raise AdapterError(f"unknown adapter {adapter!r}; supported: {sorted(ADAPTERS)}") from error
    return implementation.adapt(checkpoint)
