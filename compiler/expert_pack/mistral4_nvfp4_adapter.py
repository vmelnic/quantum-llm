"""Strict adapter for native Mistral 4 NVFP4 checkpoints."""

from __future__ import annotations

import math
import struct

from .adapters import (
    AdaptedModel,
    ExpertSource,
    HIDDEN_BATCH_ABI,
    NativeNvfp4MatrixSource,
    POSITION_BATCH_ABI,
    ROUTE_INDEX_BATCH_ABI,
    ROUTE_WEIGHT_BATCH_ABI,
    RuntimeComponentTopology,
    RuntimeLayerTopology,
    RuntimeModelTopology,
    RuntimeOperationTopology,
    TOKEN_BATCH_ABI,
)
from .constants import DTYPE_BYTES, NVFP4_QUANT_PROFILE
from .errors import AdapterError
from .safetensors import SafeTensorCheckpoint, TensorInfo


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdapterError(f"Mistral 4 requires positive integer {name}")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterError(f"Mistral 4 requires numeric {name}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise AdapterError(f"Mistral 4 requires positive finite {name}")
    return result


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _yarn_attention_scale(qk_head_dim: int, factor: float,
                          apply_scale: bool) -> float:
    """Translate the native Mistral YaRN score-scaling contract."""
    scale = qk_head_dim ** -0.5
    # Mistral's native `apply_scale=false` maps to YaRN mscale_all_dim=1.
    # The reference attention path applies the resulting mscale twice to the
    # QK score scale, independently of the RoPE inverse-frequency blend.
    if not apply_scale and factor > 1.0:
        mscale = 1.0 + 0.1 * math.log(factor)
        scale *= mscale * mscale
    return scale


class Mistral4Nvfp4Adapter:
    """Maps the native Ministral checkpoint without requantizing its organs."""

    name = "mistral4_nvfp4"

    def adapt(self, checkpoint: SafeTensorCheckpoint) -> AdaptedModel:
        config = checkpoint.config
        hidden = _positive_int(config.get("dim"), "dim")
        layers = _positive_int(config.get("n_layers"), "n_layers")
        heads = _positive_int(config.get("n_heads"), "n_heads")
        head_dim = _positive_int(config.get("head_dim"), "head_dim")
        vocab = _positive_int(config.get("vocab_size"), "vocab_size")
        maximum_context = _positive_int(
            config.get("max_position_embeddings"), "max_position_embeddings"
        )
        q_rank = _positive_int(config.get("q_lora_rank"), "q_lora_rank")
        kv_rank = _positive_int(config.get("kv_lora_rank"), "kv_lora_rank")
        qk_nope = _positive_int(
            config.get("qk_nope_head_dim"), "qk_nope_head_dim"
        )
        qk_rope = _positive_int(
            config.get("qk_rope_head_dim"), "qk_rope_head_dim"
        )
        value_dim = _positive_int(config.get("v_head_dim"), "v_head_dim")
        epsilon = _positive_float(config.get("norm_eps"), "norm_eps")
        rope_theta = _positive_float(config.get("rope_theta"), "rope_theta")
        if hidden != heads * head_dim or head_dim != qk_nope + qk_rope:
            raise AdapterError("Mistral 4 attention geometry is inconsistent")

        moe = config.get("moe")
        if not isinstance(moe, dict):
            raise AdapterError("Mistral 4 MoE config is absent")
        experts = _positive_int(moe.get("num_experts"), "moe.num_experts")
        top_k = _positive_int(
            moe.get("num_experts_per_tok"), "moe.num_experts_per_tok"
        )
        expert_width = _positive_int(
            moe.get("expert_hidden_dim"), "moe.expert_hidden_dim"
        )
        shared_count = _positive_int(
            moe.get("num_shared_experts"), "moe.num_shared_experts"
        )
        groups = _positive_int(
            moe.get("num_expert_groups"), "moe.num_expert_groups"
        )
        selected_groups = _positive_int(
            moe.get("num_expert_groups_per_tok"),
            "moe.num_expert_groups_per_tok",
        )
        route_scale = _positive_float(moe.get("routed_scale"), "routed_scale")
        if (top_k > experts or shared_count != 1 or groups != 1 or
                selected_groups != 1 or
                moe.get("first_k_dense_replace") != 0 or
                moe.get("route_every_n") != 1):
            raise AdapterError("Mistral 4 routed policy is unsupported")

        quant = config.get("quantization_config")
        groups_config = quant.get("config_groups") if isinstance(quant, dict) else None
        nvfp4 = groups_config.get("NVFP4A16") if isinstance(groups_config, dict) else None
        weights = nvfp4.get("weights") if isinstance(nvfp4, dict) else None
        activations = nvfp4.get("input_activations") if isinstance(nvfp4, dict) else None
        if (not isinstance(quant, dict) or
                quant.get("format") != "nvfp4-pack-quantized" or
                not isinstance(weights, dict) or weights.get("num_bits") != 4 or
                weights.get("group_size") != 16 or
                weights.get("scale_dtype") != "torch.float8_e4m3fn" or
                not isinstance(activations, dict) or
                activations.get("num_bits") != 4 or
                activations.get("group_size") != 16 or
                activations.get("dynamic") != "local"):
            raise AdapterError("Mistral 4 native NVFP4 contract is unsupported")

        yarn = config.get("yarn")
        llama = config.get("llama_4_scaling")
        if (not isinstance(yarn, dict) or not isinstance(llama, dict) or
                yarn.get("apply_scale") is not False):
            raise AdapterError("Mistral 4 long-context scaling is absent")
        yarn_factor = _positive_float(yarn.get("factor"), "yarn.factor")
        yarn_beta_fast = _positive_float(yarn.get("beta"), "yarn.beta")
        yarn_beta_slow = _positive_float(yarn.get("alpha"), "yarn.alpha")
        original_context = _positive_int(
            yarn.get("original_max_position_embeddings"),
            "yarn.original_max_position_embeddings",
        )
        llama_beta = _positive_float(llama.get("beta"), "llama_4_scaling.beta")
        if (_positive_int(llama.get("original_max_position_embeddings"),
                          "llama_4_scaling.original_max_position_embeddings") !=
                original_context):
            raise AdapterError("Mistral 4 long-context origins disagree")
        attention_scale = _yarn_attention_scale(
            qk_nope + qk_rope, yarn_factor, bool(yarn["apply_scale"])
        )

        tensors = checkpoint.tensors

        def require(name: str, dtype: str, shape: tuple[int, ...]) -> TensorInfo:
            info = tensors.get(name)
            if info is None or info.dtype != dtype or info.shape != shape:
                actual = None if info is None else (info.dtype, info.shape)
                raise AdapterError(
                    f"Mistral 4 tensor {name} requires {dtype} {shape}, got {actual}"
                )
            return info

        def native(prefix: str, rows: int, columns: int,
                   input_dtypes: frozenset[str] = frozenset(("F32",))) \
                -> NativeNvfp4MatrixSource:
            weight = require(prefix + ".weight_packed", "U8", (rows, columns // 2))
            scale = require(prefix + ".weight_scale", "F8_E4M3",
                            (rows, columns // 16))
            weight_global = require(prefix + ".weight_global_scale", "F32", (1,))
            input_global = tensors.get(prefix + ".input_global_scale")
            if (input_global is None or input_global.shape != (1,) or
                    input_global.dtype not in input_dtypes):
                raise AdapterError(
                    f"Mistral 4 tensor {prefix}.input_global_scale has an invalid scalar encoding"
                )
            return NativeNvfp4MatrixSource(
                weight, scale, weight_global, input_global, (rows, columns)
            )

        dense: dict[str, TensorInfo] = {}
        dense_bfloat16: set[str] = set()
        auxiliary: set[str] = set()
        dense_native: list[NativeNvfp4MatrixSource] = []

        def dense_bf16(name: str, shape: tuple[int, ...], *, aux: bool = False) -> None:
            info = require(name, "BF16", shape)
            dense[name] = info
            dense_bfloat16.add(name)
            if aux:
                auxiliary.add(name)

        dense_bf16("tok_embeddings.weight", (vocab, hidden))
        dense_bf16("output.weight", (vocab, hidden))
        dense_bf16("norm.weight", (hidden,))

        expert_sources: list[ExpertSource] = []
        for layer in range(layers):
            prefix = f"layers.{layer}."
            dense_bf16(prefix + "attention_norm.weight", (hidden,))
            dense_bf16(prefix + "ffn_norm.weight", (hidden,))
            dense_bf16(prefix + "gate.weight", (experts, hidden))
            dense_bf16(prefix + "attention.wq_a.weight", (q_rank, hidden))
            dense_bf16(prefix + "attention.q_a_norm.weight", (q_rank,))
            dense_bf16(prefix + "attention.wq_b.weight", (heads * head_dim, q_rank))
            dense_bf16(prefix + "attention.wkv_a_with_mqa.weight",
                        (kv_rank + qk_rope, hidden))
            dense_bf16(prefix + "attention.kv_a_norm.weight", (kv_rank,))
            dense_bf16(prefix + "attention.wkv_b.weight",
                        (heads * (qk_nope + value_dim), kv_rank))
            dense_bf16(prefix + "attention.wo.weight", (hidden, heads * value_dim))

            shared_prefix = prefix + "shared_experts."
            for projection, rows, columns in (
                ("w1", expert_width, hidden),
                ("w3", expert_width, hidden),
                ("w2", hidden, expert_width),
            ):
                matrix = native(
                    shared_prefix + projection, rows, columns,
                    frozenset(("F32", "BF16")),
                )
                dense[matrix.weight.name] = matrix.weight
                dense_native.append(matrix)

            for expert in range(experts):
                expert_prefix = prefix + f"experts.{expert}."
                gate = native(expert_prefix + "w1", expert_width, hidden)
                up = native(expert_prefix + "w3", expert_width, hidden)
                down = native(expert_prefix + "w2", hidden, expert_width)
                expert_sources.append(ExpertSource(
                    layer, expert, gate.weight, up.weight, down.weight,
                    gate, up, down,
                ))

        vision = config.get("vision_encoder")
        if not isinstance(vision, dict):
            raise AdapterError("Mistral 4 vision config is absent")
        vision_hidden = _positive_int(vision.get("hidden_size"), "vision.hidden_size")
        vision_layers = _positive_int(
            vision.get("num_hidden_layers"), "vision.num_hidden_layers"
        )
        vision_width = _positive_int(
            vision.get("intermediate_size"), "vision.intermediate_size"
        )
        channels = _positive_int(vision.get("num_channels"), "vision.num_channels")
        patch = _positive_int(vision.get("patch_size"), "vision.patch_size")
        dense_bf16("vision_encoder.ln_pre.weight", (vision_hidden,), aux=True)
        dense_bf16("vision_encoder.patch_conv.weight",
                    (vision_hidden, channels, patch, patch), aux=True)
        for layer in range(vision_layers):
            prefix = f"vision_encoder.transformer.layers.{layer}."
            dense_bf16(prefix + "attention_norm.weight", (vision_hidden,), aux=True)
            dense_bf16(prefix + "ffn_norm.weight", (vision_hidden,), aux=True)
            for projection in ("wq", "wk", "wv", "wo"):
                dense_bf16(prefix + f"attention.{projection}.weight",
                            (vision_hidden, vision_hidden), aux=True)
            dense_bf16(prefix + "feed_forward.w1.weight",
                        (vision_width, vision_hidden), aux=True)
            dense_bf16(prefix + "feed_forward.w3.weight",
                        (vision_width, vision_hidden), aux=True)
            dense_bf16(prefix + "feed_forward.w2.weight",
                        (vision_hidden, vision_width), aux=True)
        dense_bf16("pre_mm_projector_norm.weight", (vision_hidden,), aux=True)
        dense_bf16("patch_merger.merging_layer.weight",
                    (vision_hidden, 4 * vision_hidden), aux=True)
        dense_bf16("vision_language_adapter.w_in.weight",
                    (hidden, vision_hidden), aux=True)
        dense_bf16("vision_language_adapter.w_out.weight",
                    (hidden, hidden), aux=True)

        consumed = set(dense)
        for matrix in dense_native:
            consumed.update((matrix.weight.name, matrix.scale.name,
                             matrix.weight_global_scale.name,
                             matrix.input_global_scale.name))
        for expert in expert_sources:
            for matrix in (expert.nvfp4_gate, expert.nvfp4_up, expert.nvfp4_down):
                assert matrix is not None
                consumed.update((matrix.weight.name, matrix.scale.name,
                                 matrix.weight_global_scale.name,
                                 matrix.input_global_scale.name))
        if consumed != set(tensors):
            missing = sorted(set(tensors) - consumed)
            extra = sorted(consumed - set(tensors))
            raise AdapterError(
                f"Mistral 4 tensor partition mismatch; missing={missing[:3]}, extra={extra[:3]}"
            )

        norm_bits = _f32_bits(epsilon)
        rope_bits = _f32_bits(rope_theta)
        exact_kv_bytes = layers * (kv_rank + qk_rope) * DTYPE_BYTES["F16"]
        attributes = (
            ("attention_heads", heads), ("kv_heads", 1),
            ("head_dim", (kv_rank + qk_rope) // 2),
            ("rotary_dimension", qk_rope),
            ("full_attention_layers", layers),
            ("q_lora_rank", q_rank), ("kv_lora_rank", kv_rank),
            ("qk_nope_head_dim", qk_nope),
            ("qk_rope_head_dim", qk_rope), ("v_head_dim", value_dim),
            ("expert_count", experts), ("route_width", top_k),
            ("shared_intermediate_size", expert_width),
            ("norm_epsilon_f32_bits", norm_bits),
            ("rope_theta_f32_bits", rope_bits),
            ("rope_factor_f32_bits", _f32_bits(yarn_factor)),
            ("rope_beta_fast_f32_bits", _f32_bits(yarn_beta_fast)),
            ("rope_beta_slow_f32_bits", _f32_bits(yarn_beta_slow)),
            ("rope_original_context", original_context),
            ("llama4_scaling_beta_f32_bits", _f32_bits(llama_beta)),
            ("attention_scale_f32_bits", _f32_bits(attention_scale)),
            ("rope_interleave", 1),
            ("minimum_exact_kv_bytes_per_token", exact_kv_bytes),
            ("mtp_layers", 0),
        )
        component = RuntimeComponentTopology(
            name="decoder", layer_count=layers,
            experts_per_layer=experts, route_width=top_k,
            shared_experts_per_layer=1, hidden_size=hidden,
            intermediate_size=expert_width,
            execution_capability="moe.swiglu.routed.nvfp4-block16.merge-shared.v1",
            router_capability="router.softmax-topk.shared-swiglu.nvfp4-block16.v1",
            attributes=(("route_scale_f32_bits", _f32_bits(route_scale)),),
        )
        runtime_layers = tuple(RuntimeLayerTopology(
            logical_layer=layer,
            block_capability="block.mla.causal.latent-kv.bfloat16.v1",
            block_abi=1, routed_component="decoder", component_layer=layer,
        ) for layer in range(layers))

        operations: list[RuntimeOperationTopology] = [RuntimeOperationTopology(
            logical_layer=None, capability="embedding.lookup.bfloat16.v1",
            abi=1, routed_component=None, component_layer=0,
            tensor_bindings=(("weight", "tok_embeddings.weight"),),
            input_bindings=(("token_ids", "request.token_ids", TOKEN_BATCH_ABI),),
            output_bindings=(("hidden", "hidden.0", HIDDEN_BATCH_ABI),),
        )]
        for layer in range(layers):
            prefix = f"layers.{layer}."
            after_attention = f"layer.{layer}.after_attention"
            operations.append(RuntimeOperationTopology(
                logical_layer=layer,
                capability="block.mla.causal.latent-kv.bfloat16.v1",
                abi=1, routed_component=None, component_layer=0,
                parameters=(("norm_epsilon_f32_bits", norm_bits),),
                tensor_bindings=(
                    ("input_norm", prefix + "attention_norm.weight"),
                    ("query_a", prefix + "attention.wq_a.weight"),
                    ("query_a_norm", prefix + "attention.q_a_norm.weight"),
                    ("query_b", prefix + "attention.wq_b.weight"),
                    ("kv_a", prefix + "attention.wkv_a_with_mqa.weight"),
                    ("kv_a_norm", prefix + "attention.kv_a_norm.weight"),
                    ("kv_b", prefix + "attention.wkv_b.weight"),
                    ("output", prefix + "attention.wo.weight"),
                ),
                input_bindings=(
                    ("hidden", f"hidden.{layer}", HIDDEN_BATCH_ABI),
                    ("positions", "request.positions", POSITION_BATCH_ABI),
                ),
                output_bindings=(("hidden", after_attention, HIDDEN_BATCH_ABI),),
            ))
            expert_input = f"layer.{layer}.expert_input"
            route_indices = f"layer.{layer}.route_indices"
            route_weights = f"layer.{layer}.route_weights"
            residual = f"layer.{layer}.residual"
            shared_output = f"layer.{layer}.shared_output"
            shared = prefix + "shared_experts."
            operations.append(RuntimeOperationTopology(
                logical_layer=layer,
                capability="router.softmax-topk.shared-swiglu.nvfp4-block16.v1",
                abi=1, routed_component="decoder", component_layer=layer,
                parameters=(
                    ("norm_epsilon_f32_bits", norm_bits),
                    ("shared_intermediate_size", expert_width),
                    ("route_scale_f32_bits", _f32_bits(route_scale)),
                ),
                tensor_bindings=(
                    ("input_norm", prefix + "ffn_norm.weight"),
                    ("router_weight", prefix + "gate.weight"),
                    ("shared_gate_projection", shared + "w1.weight_packed"),
                    ("shared_up_projection", shared + "w3.weight_packed"),
                    ("shared_down_projection", shared + "w2.weight_packed"),
                ),
                input_bindings=(("hidden", after_attention, HIDDEN_BATCH_ABI),),
                output_bindings=(
                    ("expert_input", expert_input, HIDDEN_BATCH_ABI),
                    ("route_indices", route_indices, ROUTE_INDEX_BATCH_ABI),
                    ("route_weights", route_weights, ROUTE_WEIGHT_BATCH_ABI),
                    ("residual", residual, HIDDEN_BATCH_ABI),
                    ("shared_output", shared_output, HIDDEN_BATCH_ABI),
                ),
            ))
            operations.append(RuntimeOperationTopology(
                logical_layer=layer,
                capability="moe.swiglu.routed.nvfp4-block16.merge-shared.v1",
                abi=1, routed_component="decoder", component_layer=layer,
                input_bindings=(
                    ("expert_input", expert_input, HIDDEN_BATCH_ABI),
                    ("route_indices", route_indices, ROUTE_INDEX_BATCH_ABI),
                    ("route_weights", route_weights, ROUTE_WEIGHT_BATCH_ABI),
                    ("residual", residual, HIDDEN_BATCH_ABI),
                    ("shared_output", shared_output, HIDDEN_BATCH_ABI),
                ),
                output_bindings=(("hidden", f"hidden.{layer + 1}", HIDDEN_BATCH_ABI),),
            ))
        operations.append(RuntimeOperationTopology(
            logical_layer=None,
            capability="head.rmsnorm.token-select.bfloat16.v1", abi=1,
            routed_component=None, component_layer=0,
            tensor_bindings=(("norm", "norm.weight"), ("weight", "output.weight")),
            input_bindings=(("hidden", f"hidden.{layers}", HIDDEN_BATCH_ABI),),
            output_bindings=(("token_ids", "response.token_ids", TOKEN_BATCH_ABI),),
        ))

        topology = RuntimeModelTopology(
            architecture_id=self.name, vocab_size=vocab,
            max_context_tokens=maximum_context, hidden_size=hidden,
            attributes=attributes,
            required_kernels=(
                ("embedding.lookup.bfloat16.v1", 1),
                ("block.mla.causal.latent-kv.bfloat16.v1", 1),
                ("router.softmax-topk.shared-swiglu.nvfp4-block16.v1", 1),
                ("moe.swiglu.routed.nvfp4-block16.merge-shared.v1", 1),
                ("head.rmsnorm.token-select.bfloat16.v1", 1),
            ),
            components=(component,), layers=runtime_layers,
            operations=tuple(operations),
            tensor_bindings=(
                ("token_embedding", "tok_embeddings.weight"),
                ("final_norm", "norm.weight"),
                ("output_head", "output.weight"),
            ),
            program_inputs=(
                ("token_ids", "request.token_ids", TOKEN_BATCH_ABI),
                ("positions", "request.positions", POSITION_BATCH_ABI),
            ),
            program_outputs=(
                ("next_token_ids", "response.token_ids", TOKEN_BATCH_ABI),
            ),
        )
        architecture = {
            "model_type": self.name,
            "hidden_size": hidden,
            "intermediate_size": expert_width,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": heads,
            "num_experts": experts,
            "experts_per_token": top_k,
            "hidden_activation": "silu",
            "maximum_context_tokens": maximum_context,
            "native_quantization": NVFP4_QUANT_PROFILE,
        }
        return AdaptedModel(
            family=self.name, architecture=architecture,
            dense=tuple(dense[name] for name in sorted(dense)),
            experts=tuple(expert_sources), source_tensor_count=len(tensors),
            runtime_topology=topology,
            dense_nvfp4=tuple(dense_native),
            dense_bfloat16=frozenset(dense_bfloat16),
            auxiliary_dense=frozenset(auxiliary),
            template_reasoning_effort_map=(
                ("off", "none"), ("low", "high"),
                ("medium", "high"), ("xhigh", "high"),
            ),
            supported_expert_quant_profiles=frozenset((NVFP4_QUANT_PROFILE,)),
        )
