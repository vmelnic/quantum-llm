"""Strict source adapter for the Qwen4 experimental hybrid-MoE ABI.

The upstream name is intentionally confined to this source adapter.  The
published program is expressed only in geometry-driven VM capabilities.
"""

from __future__ import annotations

import math

from .adapters import (
    HIDDEN_BATCH_ABI,
    POSITION_BATCH_ABI,
    ROUTE_INDEX_BATCH_ABI,
    ROUTE_WEIGHT_BATCH_ABI,
    TOKEN_BATCH_ABI,
    AdaptedModel,
    ExpertSource,
    RuntimeComponentTopology,
    RuntimeLayerTopology,
    RuntimeModelTopology,
    RuntimeOperationTopology,
    _float32_bits,
    _integer,
    _number,
)
from .constants import DTYPE_BYTES, FP4_QUANT_PROFILES
from .errors import AdapterError
from .safetensors import SafeTensorCheckpoint


HYPER_BATCH_ABI = "batch.hyper-hidden.f32.cuda.v1"
INJECTION_BATCH_ABI = "batch.hyper-injection.f32.cuda.v1"


def _prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def _nth_primes_after(start: int, count: int) -> list[int]:
    result: list[int] = []
    value = start
    while len(result) < count:
        value += 1
        if _prime(value):
            result.append(value)
    return result


class Qwen4ExpAdapter:
    """Map the upstream checkpoint to generic Hyper/QSA/PLE VM operations."""

    name = "qwen4_exp"

    def adapt(self, checkpoint: SafeTensorCheckpoint) -> AdaptedModel:
        config = checkpoint.config
        if config.get("model_type") != "qwen4_exp":
            raise AdapterError("qwen4_exp adapter requires model_type qwen4_exp")
        if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
            raise AdapterError("qwen4_exp conditional-generation architecture is required")
        text = config.get("text_config")
        vision = config.get("vision_config")
        if not isinstance(text, dict) or not isinstance(vision, dict):
            raise AdapterError("qwen4_exp text/vision configs are absent")
        if text.get("model_type") != "qwen4_exp_text":
            raise AdapterError("qwen4_exp text_config type is unsupported")

        layers = _integer(text, "num_hidden_layers")
        hidden = _integer(text, "hidden_size")
        hyper_count = _integer(text, "hc_count")
        hyper_width = hidden * hyper_count
        hyper_lowrank = _integer(text, "hc_lowrank")
        expert_width = _integer(text, "moe_intermediate_size")
        shared_width = _integer(text, "shared_expert_intermediate_size")
        experts = _integer(text, "num_experts")
        top_k = _integer(text, "num_experts_per_tok")
        vocab = _integer(text, "vocab_size")
        max_context = _integer(text, "max_position_embeddings")
        heads = _integer(text, "num_attention_heads")
        kv_heads = _integer(text, "num_key_value_heads")
        head_dim = _integer(text, "head_dim")
        full_interval = _integer(text, "full_attention_interval")
        key_heads = _integer(text, "linear_num_key_heads")
        value_heads = _integer(text, "linear_num_value_heads")
        key_head_dim = _integer(text, "linear_key_head_dim")
        value_head_dim = _integer(text, "linear_value_head_dim")
        conv_kernel = _integer(text, "linear_conv_kernel_dim")
        index_heads = _integer(text, "indexer_n_heads")
        index_kv_heads = _integer(text, "indexer_kv_heads")
        index_head_dim = _integer(text, "indexer_head_dim")
        index_budget = _integer(text, "indexer_budget")
        index_compress = _integer(text, "indexer_compress_ratio")
        ngram_size = _integer(text, "ngram_size")
        heads_per_ngram = _integer(text, "heads_per_ngram")
        ple_width = _integer(text, "ple_embed_dim")
        ple_conv = _integer(text, "ple_conv_kernel_size")
        split_parts = _integer(text, "split_ngram_parts")
        ngram_base = _integer(text, "ngram_vocab_size_base")
        ngram_divisor = _integer(text, "make_ngram_vocab_size_divisible_by")
        epsilon = _number(text, "rms_norm_eps")
        layer_types = text.get("layer_types")
        ple_layer_ids = text.get("ple_layer_ids")
        rope = text.get("rope_parameters")
        if (
            top_k > experts
            or heads % kv_heads
            or value_heads % key_heads
            or index_kv_heads != 1
            or index_budget % index_compress
            or ple_width % ((ngram_size - 1) * heads_per_ngram)
            or not isinstance(layer_types, list)
            or len(layer_types) != layers
            or any(item not in {"linear_attention", "full_attention"} for item in layer_types)
            or any(
                (item == "full_attention") != ((layer + 1) % full_interval == 0)
                for layer, item in enumerate(layer_types)
            )
            or not isinstance(ple_layer_ids, list)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 or item > layers
                   for item in ple_layer_ids)
            or not isinstance(rope, dict)
        ):
            raise AdapterError("qwen4_exp text geometry is inconsistent")
        partial_rotary = rope.get("partial_rotary_factor")
        if isinstance(partial_rotary, bool) or not isinstance(partial_rotary, (int, float)):
            raise AdapterError("qwen4_exp partial rotary factor is invalid")
        rotary_dim = int(head_dim * float(partial_rotary))
        if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > head_dim:
            raise AdapterError("qwen4_exp rotary dimension is invalid")
        rope_theta = _number(rope, "rope_theta")
        output_gate_type = text.get("output_gate_type") or text.get("hidden_act")
        output_gate_activation = {"silu": 1, "sigmoid": 2}.get(
            output_gate_type
        )
        if output_gate_activation is None:
            raise AdapterError("qwen4_exp recurrent output gate is unsupported")
        eos = text.get("eos_token_id")
        if isinstance(eos, list):
            eos = eos[0] if len(eos) == 1 else None
        if isinstance(eos, bool) or not isinstance(eos, int) or eos < 0:
            raise AdapterError("qwen4_exp EOS token is invalid")

        expected: dict[str, tuple[int, ...]] = {}
        fp4: set[str] = set()
        float32: set[str] = set()
        int64: set[str] = set()
        auxiliary: set[str] = set()
        host_mapped: set[str] = set()

        def add(
            name: str,
            shape: tuple[int, ...],
            *,
            matrix: bool = False,
            exact_control: bool = False,
            raw_i64: bool = False,
            aux: bool = False,
            host: bool = False,
        ) -> None:
            if (
                name in expected
                or sum((matrix, raw_i64)) > 1
                or (exact_control and not matrix)
            ):
                raise AdapterError(f"duplicate/ambiguous qwen4_exp tensor {name}")
            expected[name] = shape
            # The artifact's routed expert pages use E2M1 FP4.  Rank-two
            # control/state matrices and sparse PLE tables deliberately use
            # the existing per-row INT8 ABI: reducing those organs to FP4
            # corrupts routing/state semantics on long prompts.  Higher-rank
            # convolution payloads retain FP4 because the INT8 row ABI is
            # rank-two by contract.
            # Small recurrent and Hyper control projections are compounded at
            # every layer.  Their measured per-row INT8 error is materially
            # higher than the large data projections, so the source adapter
            # declares them as exact controls while the runtime remains
            # geometry/ABI driven.
            if exact_control:
                float32.add(name)
            elif matrix:
                if len(shape) != 2:
                    fp4.add(name)
            elif raw_i64:
                int64.add(name)
            else:
                float32.add(name)
            if aux:
                auxiliary.add(name)
            if host:
                host_mapped.add(name)

        prefix = "model.language_model."
        add(prefix + "embed_tokens.weight", (vocab, hidden), matrix=True)
        add("lm_head.weight", (vocab, hidden), matrix=True)
        add(prefix + "hyper_connection_mixer.hc_norm.weight", (hyper_width,))
        add(prefix + "hyper_connection_mixer.input_mix_weight_down.weight",
            (hyper_lowrank, hyper_width), matrix=True)
        add(prefix + "hyper_connection_mixer.input_mix_weight_up.weight",
            (hyper_width, hyper_lowrank), matrix=True)

        key_dim = key_heads * key_head_dim
        value_dim = value_heads * value_head_dim
        conv_dim = 2 * key_dim + value_dim
        routed_physical: set[str] = set()
        for layer, layer_type in enumerate(layer_types):
            layer_prefix = f"{prefix}layers.{layer}."
            for organ in ("attn_hyper_connection", "mlp_hyper_connection"):
                organ_prefix = layer_prefix + organ + "."
                add(organ_prefix + "hc_norm.weight", (hyper_width,))
                add(organ_prefix + "input_mix_weight_down.weight",
                    (hyper_lowrank, hyper_width), matrix=True)
                add(organ_prefix + "input_mix_weight_up.weight",
                    (hyper_width, hyper_lowrank), matrix=True)
                add(organ_prefix + "block_inject_weight.weight",
                    (hyper_count, hyper_width), matrix=True,
                    exact_control=True)
            if layer_type == "linear_attention":
                linear = layer_prefix + "linear_attn."
                add(linear + "in_proj_qkv.weight", (conv_dim, hidden), matrix=True)
                add(linear + "in_proj_z.weight", (value_dim, hidden), matrix=True)
                add(linear + "in_proj_b.weight", (value_heads, hidden),
                    matrix=True, exact_control=True)
                add(linear + "in_proj_a.weight", (value_heads, hidden),
                    matrix=True, exact_control=True)
                add(linear + "conv1d.weight", (conv_dim, 1, conv_kernel), matrix=True)
                add(linear + "dt_bias", (value_heads,))
                add(linear + "A_log", (value_heads,))
                add(linear + "norm.weight", (value_head_dim,))
                add(linear + "out_proj.weight", (hidden, value_dim), matrix=True)
            else:
                attention = layer_prefix + "self_attn."
                add(attention + "q_proj.weight", (2 * heads * head_dim, hidden), matrix=True)
                add(attention + "k_proj.weight", (kv_heads * head_dim, hidden), matrix=True)
                add(attention + "v_proj.weight", (kv_heads * head_dim, hidden), matrix=True)
                add(attention + "o_proj.weight", (hidden, heads * head_dim), matrix=True)
                add(attention + "q_norm.weight", (head_dim,))
                add(attention + "k_norm.weight", (head_dim,))
                add(attention + "indexer.index_qk_proj.weight",
                    ((index_heads + index_kv_heads) * index_head_dim, hidden), matrix=True)
                add(attention + "indexer.q_layernorm.weight", (index_head_dim,))
                add(attention + "indexer.k_layernorm.weight", (index_head_dim,))

            mlp = layer_prefix + "mlp."
            add(mlp + "gate.weight", (experts, hidden))
            add(mlp + "shared_expert.gate_proj.weight", (shared_width, hidden), matrix=True)
            add(mlp + "shared_expert.up_proj.weight", (shared_width, hidden), matrix=True)
            add(mlp + "shared_expert.down_proj.weight", (hidden, shared_width), matrix=True)
            add(mlp + "shared_expert_gate.weight", (1, hidden))
            routed_physical.update({mlp + "experts.gate_up_proj", mlp + "experts.down_proj"})

        ngram_heads = (ngram_size - 1) * heads_per_ngram
        primes = _nth_primes_after(ngram_base - 1, ngram_heads)
        total_ngram = sum(primes)
        padded_ngram = math.ceil(total_ngram / ngram_divisor) * ngram_divisor
        if padded_ngram % split_parts:
            raise AdapterError("qwen4_exp ngram vocabulary is not shard-divisible")
        rows_per_shard = padded_ngram // split_parts
        per_head_width = ple_width // ngram_heads
        for one_based_layer in ple_layer_ids:
            ple_prefix = f"{prefix}layers.{one_based_layer - 1}.ple."
            add(ple_prefix + "key_proj.weight", (hyper_width, ple_width), matrix=True)
            add(ple_prefix + "value_proj.weight", (hidden, ple_width), matrix=True)
            for role in ("norm_key.weight", "norm_query.weight", "norm_conv.weight"):
                add(ple_prefix + role, (hyper_width,))
            add(ple_prefix + "conv1d.weight", (hyper_width, 1, ple_conv), matrix=True)
            embedding = ple_prefix + "ple_embedding."
            for role, shape in (
                ("layer_multipliers", (ngram_size,)),
                ("ngram_heads_vocab_sizes", (ngram_heads,)),
                ("ngram_heads_offsets", (ngram_heads,)),
            ):
                add(embedding + role, shape, raw_i64=True, host=True)
            for shard in range(split_parts):
                add(embedding + f"ngram_embedding.shard_{shard}.weight",
                    (rows_per_shard, per_head_width), matrix=True, host=True)

        vision_layers = _integer(vision, "depth")
        vision_hidden = _integer(vision, "hidden_size")
        vision_intermediate = _integer(vision, "intermediate_size")
        vision_positions = _integer(vision, "num_position_embeddings")
        vision_channels = _integer(vision, "in_channels")
        patch = _integer(vision, "patch_size")
        temporal_patch = _integer(vision, "temporal_patch_size")
        merge = _integer(vision, "spatial_merge_size")
        vision_output = _integer(vision, "out_hidden_size")
        if vision_output != hidden:
            raise AdapterError("qwen4_exp vision output disagrees with text hidden size")
        visual = "model.visual."
        add(visual + "patch_embed.proj.weight",
            (vision_hidden, vision_channels, temporal_patch, patch, patch), matrix=True, aux=True)
        add(visual + "patch_embed.proj.bias", (vision_hidden,), aux=True)
        add(visual + "pos_embed.weight", (vision_positions, vision_hidden), matrix=True, aux=True)
        for layer in range(vision_layers):
            block = f"{visual}blocks.{layer}."
            for norm in ("norm1", "norm2"):
                add(block + norm + ".weight", (vision_hidden,), aux=True)
                add(block + norm + ".bias", (vision_hidden,), aux=True)
            add(block + "attn.qkv.weight", (3 * vision_hidden, vision_hidden), matrix=True, aux=True)
            add(block + "attn.qkv.bias", (3 * vision_hidden,), aux=True)
            add(block + "attn.proj.weight", (vision_hidden, vision_hidden), matrix=True, aux=True)
            add(block + "attn.proj.bias", (vision_hidden,), aux=True)
            add(block + "mlp.linear_fc1.weight",
                (vision_intermediate, vision_hidden), matrix=True, aux=True)
            add(block + "mlp.linear_fc1.bias", (vision_intermediate,), aux=True)
            add(block + "mlp.linear_fc2.weight",
                (vision_hidden, vision_intermediate), matrix=True, aux=True)
            add(block + "mlp.linear_fc2.bias", (vision_hidden,), aux=True)
        merged = vision_hidden * merge * merge
        add(visual + "merger.norm.weight", (vision_hidden,), aux=True)
        add(visual + "merger.norm.bias", (vision_hidden,), aux=True)
        add(visual + "merger.linear_fc1.weight", (merged, merged), matrix=True, aux=True)
        add(visual + "merger.linear_fc1.bias", (merged,), aux=True)
        add(visual + "merger.linear_fc2.weight", (hidden, merged), matrix=True, aux=True)
        add(visual + "merger.linear_fc2.bias", (hidden,), aux=True)

        mtp_layers = _integer(text, "mtp_num_hidden_layers")
        if mtp_layers != 1:
            raise AdapterError("qwen4_exp adapter preserves exactly one declared MTP layer")
        for name, shape, matrix in (
            ("mtp.fc_embedding.weight", (hidden, hidden), True),
            ("mtp.fc_hidden.weight", (hidden, hidden), True),
            ("mtp.pre_fc_norm_embedding.weight", (hidden,), False),
            ("mtp.pre_fc_norm_hidden.weight", (hyper_width,), False),
            ("mtp.hyper_connection_mixer.hc_norm.weight", (hyper_width,), False),
            ("mtp.hyper_connection_mixer.input_mix_weight_down.weight", (hyper_lowrank, hyper_width), True),
            ("mtp.hyper_connection_mixer.input_mix_weight_up.weight", (hyper_width, hyper_lowrank), True),
        ):
            add(name, shape, matrix=matrix, aux=True)
        mtp = "mtp.layers.0."
        for organ in ("attn_hyper_connection", "mlp_hyper_connection"):
            organ_prefix = mtp + organ + "."
            add(organ_prefix + "hc_norm.weight", (hyper_width,), aux=True)
            add(organ_prefix + "input_mix_weight_down.weight",
                (hyper_lowrank, hyper_width), matrix=True, aux=True)
            add(organ_prefix + "input_mix_weight_up.weight",
                (hyper_width, hyper_lowrank), matrix=True, aux=True)
            add(organ_prefix + "block_inject_weight.weight",
                (hyper_count, hyper_width), matrix=True, aux=True)
        attention = mtp + "self_attn."
        for name, shape, matrix in (
            ("q_proj.weight", (2 * heads * head_dim, hidden), True),
            ("k_proj.weight", (kv_heads * head_dim, hidden), True),
            ("v_proj.weight", (kv_heads * head_dim, hidden), True),
            ("o_proj.weight", (hidden, heads * head_dim), True),
            ("q_norm.weight", (head_dim,), False),
            ("k_norm.weight", (head_dim,), False),
            ("indexer.index_qk_proj.weight", ((index_heads + index_kv_heads) * index_head_dim, hidden), True),
            ("indexer.q_layernorm.weight", (index_head_dim,), False),
            ("indexer.k_layernorm.weight", (index_head_dim,), False),
        ):
            add(attention + name, shape, matrix=matrix, aux=True)
        mtp_mlp = mtp + "mlp."
        add(mtp_mlp + "gate.weight", (experts, hidden), aux=True)
        add(mtp_mlp + "shared_expert.gate_proj.weight", (shared_width, hidden), matrix=True, aux=True)
        add(mtp_mlp + "shared_expert.up_proj.weight", (shared_width, hidden), matrix=True, aux=True)
        add(mtp_mlp + "shared_expert.down_proj.weight", (hidden, shared_width), matrix=True, aux=True)
        add(mtp_mlp + "shared_expert_gate.weight", (1, hidden), aux=True)
        add(mtp_mlp + "experts.gate_up_proj", (experts, 2 * expert_width, hidden), matrix=True, aux=True)
        add(mtp_mlp + "experts.down_proj", (experts, hidden, expert_width), matrix=True, aux=True)

        actual = set(checkpoint.tensors)
        classified = set(expected) | routed_physical
        if actual != classified:
            missing = sorted(classified - actual)
            unknown = sorted(actual - classified)
            raise AdapterError(
                "qwen4_exp tensor partition mismatch; "
                f"missing={missing[:8]}, unknown={unknown[:8]}"
            )
        for name, shape in expected.items():
            info = checkpoint.tensors[name]
            expected_dtype = "I64" if name in int64 else "BF16"
            if info.shape != shape or info.dtype != expected_dtype:
                raise AdapterError(
                    f"qwen4_exp tensor mismatch for {name}: expected "
                    f"{expected_dtype} {shape}, got {info.dtype} {info.shape}"
                )
        for layer in range(layers):
            mlp = f"{prefix}layers.{layer}.mlp.experts."
            for name, shape in (
                (mlp + "gate_up_proj", (experts, 2 * expert_width, hidden)),
                (mlp + "down_proj", (experts, hidden, expert_width)),
            ):
                info = checkpoint.tensors[name]
                if info.shape != shape or info.dtype != "BF16":
                    raise AdapterError(f"qwen4_exp routed tensor mismatch for {name}")

        routed_sources: list[ExpertSource] = []
        source_bytes = DTYPE_BYTES["BF16"]
        for layer in range(layers):
            physical_prefix = f"{prefix}layers.{layer}.mlp.experts."
            gate_up_name = physical_prefix + "gate_up_proj"
            down_name = physical_prefix + "down_proj"
            gate_up_stride = 2 * expert_width * hidden * source_bytes
            down_stride = hidden * expert_width * source_bytes
            for expert in range(experts):
                logical = f"decoder.layers.{layer}.experts.{expert}."
                base = expert * gate_up_stride
                gate = checkpoint.tensor_region(
                    gate_up_name, logical + "gate", (expert_width, hidden), base)
                up = checkpoint.tensor_region(
                    gate_up_name, logical + "up", (expert_width, hidden),
                    base + expert_width * hidden * source_bytes)
                down = checkpoint.tensor_region(
                    down_name, logical + "down", (hidden, expert_width),
                    expert * down_stride)
                routed_sources.append(ExpertSource(layer, expert, gate, up, down))

        norm_bits = _float32_bits(epsilon)
        rope_bits = _float32_bits(rope_theta)
        full_attention_layers = sum(
            item == "full_attention" for item in layer_types
        )
        # The callable provider stores exact FP16 K/V plus one exact FP16 QSA
        # index key for every populated token of every sparse-attention layer.
        # Publishing this geometry in the authenticated runtime program lets
        # deployment size paged KV without a model-name branch.
        exact_kv_bytes_per_token = full_attention_layers * (
            2 * kv_heads * head_dim + index_kv_heads * index_head_dim
        ) * DTYPE_BYTES["F16"]
        attributes = (
            ("attention_heads", heads), ("kv_heads", kv_heads),
            ("head_dim", head_dim), ("rotary_dimension", rotary_dim),
            ("full_attention_layers", full_attention_layers),
            ("minimum_exact_kv_bytes_per_token", exact_kv_bytes_per_token),
            ("linear_key_heads", key_heads), ("linear_value_heads", value_heads),
            ("linear_key_head_dim", key_head_dim),
            ("linear_value_head_dim", value_head_dim),
            ("linear_conv_kernel", conv_kernel),
            ("shared_intermediate_size", shared_width),
            ("expert_count", experts), ("route_width", top_k),
            ("norm_epsilon_f32_bits", norm_bits),
            ("rope_theta_f32_bits", rope_bits), ("zero_centered_norm", 1),
            ("mtp_layers", 0), ("hyper_connection_count", hyper_count),
            ("hyper_connection_width", hyper_width),
            ("hyper_connection_lowrank", hyper_lowrank),
            ("qsa_index_heads", index_heads),
            ("qsa_index_kv_heads", index_kv_heads),
            ("qsa_index_head_dim", index_head_dim),
            ("qsa_token_budget", index_budget),
            ("qsa_compress_ratio", index_compress),
            ("ple_ngram_size", ngram_size),
            ("ple_heads_per_ngram", heads_per_ngram),
            ("ple_embedding_width", ple_width),
            ("ple_convolution_kernel", ple_conv),
            ("ple_shard_count", split_parts),
            ("ple_rows_per_shard", rows_per_shard),
            ("ple_eos_token_id", eos),
        )
        component = RuntimeComponentTopology(
            name="decoder", layer_count=layers,
            experts_per_layer=experts, route_width=top_k,
            shared_experts_per_layer=1, hidden_size=hidden,
            intermediate_size=expert_width,
            execution_capability="moe.swiglu.routed.merge-shared.no-residual.v1",
            router_capability="router.linear-topk.shared-swiglu.no-residual.v1",
        )
        runtime_layers = tuple(RuntimeLayerTopology(
            logical_layer=layer,
            block_capability=(
                "block.sparse-attention.qsa.output-gated.v1"
                if layer_types[layer] == "full_attention"
                else "block.recurrent-linear-attention.split-gated-delta.no-residual.v1"
            ),
            block_abi=(1 if layer_types[layer] == "full_attention" else 2),
            routed_component="decoder", component_layer=layer,
        ) for layer in range(layers))

        operations: list[RuntimeOperationTopology] = [
            RuntimeOperationTopology(
                logical_layer=None, capability="embedding.lookup.v1",
                abi=1, routed_component=None, component_layer=0,
                tensor_bindings=(("weight", prefix + "embed_tokens.weight"),),
                input_bindings=(("token_ids", "request.token_ids", TOKEN_BATCH_ABI),),
                output_bindings=(("hidden", "embedding.hidden", HIDDEN_BATCH_ABI),),
            ),
            RuntimeOperationTopology(
                logical_layer=None, capability="state.hyper-connection.initialize.v1",
                abi=1, routed_component=None, component_layer=0,
                parameters=(("stream_count", hyper_count),),
                input_bindings=(("hidden", "embedding.hidden", HIDDEN_BATCH_ABI),),
                output_bindings=(("hyper", "hyper.0", HYPER_BATCH_ABI),),
            ),
        ]
        current_hyper = "hyper.0"
        for layer, layer_type in enumerate(layer_types):
            layer_prefix = f"{prefix}layers.{layer}."
            if layer + 1 in ple_layer_ids:
                ple_prefix = layer_prefix + "ple."
                bindings = [
                    ("key_projection", ple_prefix + "key_proj.weight"),
                    ("value_projection", ple_prefix + "value_proj.weight"),
                    ("key_norm", ple_prefix + "norm_key.weight"),
                    ("query_norm", ple_prefix + "norm_query.weight"),
                    ("convolution_norm", ple_prefix + "norm_conv.weight"),
                    ("convolution", ple_prefix + "conv1d.weight"),
                    ("layer_multipliers", ple_prefix + "ple_embedding.layer_multipliers"),
                    ("head_vocab_sizes", ple_prefix + "ple_embedding.ngram_heads_vocab_sizes"),
                    ("head_offsets", ple_prefix + "ple_embedding.ngram_heads_offsets"),
                ]
                bindings.extend(
                    (f"embedding_shard.{shard}",
                     ple_prefix + f"ple_embedding.ngram_embedding.shard_{shard}.weight")
                    for shard in range(split_parts)
                )
                next_hyper = f"layer.{layer}.after_ple"
                operations.append(RuntimeOperationTopology(
                    logical_layer=layer,
                    capability="embedding.ngram-ple.v1", abi=1,
                    routed_component=None, component_layer=0,
                    tensor_bindings=tuple(bindings),
                    input_bindings=(
                        ("hyper", current_hyper, HYPER_BATCH_ABI),
                        ("token_ids", "request.token_ids", TOKEN_BATCH_ABI),
                    ),
                    output_bindings=(("hyper", next_hyper, HYPER_BATCH_ABI),),
                ))
                current_hyper = next_hyper

            def hyper_read(organ: str, source: str, stem: str) -> tuple[str, str, str]:
                hidden_value = stem + ".hidden"
                retained_value = stem + ".retained"
                injection_value = stem + ".injection"
                organ_prefix = layer_prefix + organ + "."
                operations.append(RuntimeOperationTopology(
                    logical_layer=layer, capability="state.hyper-connection.read.v1",
                    abi=1, routed_component=None, component_layer=0,
                    parameters=(("stream_count", hyper_count),
                                ("norm_epsilon_f32_bits", norm_bits)),
                    tensor_bindings=(
                        ("norm", organ_prefix + "hc_norm.weight"),
                        ("mix_down", organ_prefix + "input_mix_weight_down.weight"),
                        ("mix_up", organ_prefix + "input_mix_weight_up.weight"),
                        ("inject", organ_prefix + "block_inject_weight.weight"),
                    ),
                    input_bindings=(("hyper", source, HYPER_BATCH_ABI),),
                    output_bindings=(
                        ("hidden", hidden_value, HIDDEN_BATCH_ABI),
                        ("retained", retained_value, HYPER_BATCH_ABI),
                        ("injection", injection_value, INJECTION_BATCH_ABI),
                    ),
                ))
                return hidden_value, retained_value, injection_value

            attention_input, retained, injection = hyper_read(
                "attn_hyper_connection", current_hyper,
                f"layer.{layer}.attention_read")
            block_output = f"layer.{layer}.attention_output"
            if layer_type == "full_attention":
                attention = layer_prefix + "self_attn."
                block_bindings = (
                    ("query_projection", attention + "q_proj.weight"),
                    ("key_projection", attention + "k_proj.weight"),
                    ("value_projection", attention + "v_proj.weight"),
                    ("output_projection", attention + "o_proj.weight"),
                    ("query_norm", attention + "q_norm.weight"),
                    ("key_norm", attention + "k_norm.weight"),
                    ("index_projection", attention + "indexer.index_qk_proj.weight"),
                    ("index_query_norm", attention + "indexer.q_layernorm.weight"),
                    ("index_key_norm", attention + "indexer.k_layernorm.weight"),
                )
            else:
                linear = layer_prefix + "linear_attn."
                block_bindings = (
                    ("qkv_projection", linear + "in_proj_qkv.weight"),
                    ("z_projection", linear + "in_proj_z.weight"),
                    ("b_projection", linear + "in_proj_b.weight"),
                    ("a_projection", linear + "in_proj_a.weight"),
                    ("convolution", linear + "conv1d.weight"),
                    ("time_bias", linear + "dt_bias"),
                    ("decay_log", linear + "A_log"),
                    ("output_norm", linear + "norm.weight"),
                    ("output_projection", linear + "out_proj.weight"),
                )
            operations.append(RuntimeOperationTopology(
                logical_layer=layer, capability=runtime_layers[layer].block_capability,
                abi=(1 if layer_type == "full_attention" else 2),
                routed_component=None, component_layer=0,
                parameters=(
                    ("norm_epsilon_f32_bits", norm_bits),
                    ("rope_theta_f32_bits", rope_bits),
                    ("rotary_dimension", rotary_dim),
                ) + (
                    (("output_gate_activation", output_gate_activation),)
                    if layer_type == "linear_attention" else ()
                ),
                tensor_bindings=block_bindings,
                input_bindings=(
                    ("hidden", attention_input, HIDDEN_BATCH_ABI),
                    ("positions", "request.positions", POSITION_BATCH_ABI),
                ),
                output_bindings=(("hidden", block_output, HIDDEN_BATCH_ABI),),
            ))
            after_attention = f"layer.{layer}.after_attention"
            operations.append(RuntimeOperationTopology(
                logical_layer=layer, capability="state.hyper-connection.inject.v1",
                abi=1, routed_component=None, component_layer=0,
                parameters=(("stream_count", hyper_count),),
                input_bindings=(
                    ("retained", retained, HYPER_BATCH_ABI),
                    ("hidden", block_output, HIDDEN_BATCH_ABI),
                    ("injection", injection, INJECTION_BATCH_ABI),
                ),
                output_bindings=(("hyper", after_attention, HYPER_BATCH_ABI),),
            ))
            mlp_input, retained, injection = hyper_read(
                "mlp_hyper_connection", after_attention,
                f"layer.{layer}.mlp_read")
            expert_input = f"layer.{layer}.expert_input"
            route_indices = f"layer.{layer}.route_indices"
            route_weights = f"layer.{layer}.route_weights"
            shared_output = f"layer.{layer}.shared_output"
            mlp = layer_prefix + "mlp."
            operations.append(RuntimeOperationTopology(
                logical_layer=layer,
                capability="router.linear-topk.shared-swiglu.no-residual.v1",
                abi=1, routed_component="decoder", component_layer=layer,
                parameters=(("shared_intermediate_size", shared_width),),
                tensor_bindings=(
                    ("router_weight", mlp + "gate.weight"),
                    ("shared_gate_projection", mlp + "shared_expert.gate_proj.weight"),
                    ("shared_up_projection", mlp + "shared_expert.up_proj.weight"),
                    ("shared_down_projection", mlp + "shared_expert.down_proj.weight"),
                    ("shared_router", mlp + "shared_expert_gate.weight"),
                ),
                input_bindings=(("hidden", mlp_input, HIDDEN_BATCH_ABI),),
                output_bindings=(
                    ("expert_input", expert_input, HIDDEN_BATCH_ABI),
                    ("route_indices", route_indices, ROUTE_INDEX_BATCH_ABI),
                    ("route_weights", route_weights, ROUTE_WEIGHT_BATCH_ABI),
                    ("shared_output", shared_output, HIDDEN_BATCH_ABI),
                ),
            ))
            mlp_output = f"layer.{layer}.mlp_output"
            operations.append(RuntimeOperationTopology(
                logical_layer=layer,
                capability="moe.swiglu.routed.merge-shared.no-residual.v1",
                abi=1, routed_component="decoder", component_layer=layer,
                input_bindings=(
                    ("expert_input", expert_input, HIDDEN_BATCH_ABI),
                    ("route_indices", route_indices, ROUTE_INDEX_BATCH_ABI),
                    ("route_weights", route_weights, ROUTE_WEIGHT_BATCH_ABI),
                    ("shared_output", shared_output, HIDDEN_BATCH_ABI),
                ),
                output_bindings=(("hidden", mlp_output, HIDDEN_BATCH_ABI),),
            ))
            current_hyper = f"hyper.{layer + 1}"
            operations.append(RuntimeOperationTopology(
                logical_layer=layer, capability="state.hyper-connection.inject.v1",
                abi=1, routed_component=None, component_layer=0,
                parameters=(("stream_count", hyper_count),),
                input_bindings=(
                    ("retained", retained, HYPER_BATCH_ABI),
                    ("hidden", mlp_output, HIDDEN_BATCH_ABI),
                    ("injection", injection, INJECTION_BATCH_ABI),
                ),
                output_bindings=(("hyper", current_hyper, HYPER_BATCH_ABI),),
            ))

        operations.append(RuntimeOperationTopology(
            logical_layer=None, capability="state.hyper-connection.reduce.v1",
            abi=1, routed_component=None, component_layer=0,
            parameters=(("stream_count", hyper_count),
                        ("norm_epsilon_f32_bits", norm_bits)),
            tensor_bindings=(
                ("norm", prefix + "hyper_connection_mixer.hc_norm.weight"),
                ("mix_down", prefix + "hyper_connection_mixer.input_mix_weight_down.weight"),
                ("mix_up", prefix + "hyper_connection_mixer.input_mix_weight_up.weight"),
            ),
            input_bindings=(("hyper", current_hyper, HYPER_BATCH_ABI),),
            output_bindings=(("hidden", "final.hidden", HIDDEN_BATCH_ABI),),
        ))
        operations.append(RuntimeOperationTopology(
            logical_layer=None, capability="head.token-select.no-norm.v1",
            abi=1, routed_component=None, component_layer=0,
            tensor_bindings=(("weight", "lm_head.weight"),),
            input_bindings=(("hidden", "final.hidden", HIDDEN_BATCH_ABI),),
            output_bindings=(("token_ids", "response.token_ids", TOKEN_BATCH_ABI),),
        ))

        architecture = {
            "family": self.name,
            "upstream_model_type": config["model_type"],
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
            "hidden_activation": text.get("hidden_act"),
            "attention_bias": bool(text.get("attention_bias", False)),
            "attention_dropout": text.get("attention_dropout", 0.0),
            "rms_norm_epsilon": epsilon,
            "rope": {"theta": rope_theta, "scaling": None},
            "normalize_topk_probability": bool(text.get("norm_topk_prob", True)),
            "tie_word_embeddings": bool(text.get("tie_word_embeddings", False)),
            "full_attention_interval": full_interval,
            "partial_rotary_factor": float(partial_rotary),
            "hyper_connection_count": hyper_count,
            "qsa_token_budget": index_budget,
            "qsa_compress_ratio": index_compress,
            "ple_layer_ids": list(ple_layer_ids),
            "vision_auxiliary_only": True,
            "mtp_auxiliary_only": True,
        }
        required = tuple(dict.fromkeys(
            (operation.capability, operation.abi) for operation in operations
        ))
        topology = RuntimeModelTopology(
            architecture_id="hybrid-hyper-qsa-ple-moe-v1",
            vocab_size=vocab, max_context_tokens=max_context,
            hidden_size=hidden, attributes=attributes,
            required_kernels=required, components=(component,),
            layers=runtime_layers, operations=tuple(operations),
            tensor_bindings=(
                ("token_embedding", prefix + "embed_tokens.weight"),
                ("output_head", "lm_head.weight"),
            ),
            program_inputs=(
                ("token_ids", "request.token_ids", TOKEN_BATCH_ABI),
                ("positions", "request.positions", POSITION_BATCH_ABI),
            ),
            program_outputs=(("next_token_ids", "response.token_ids", TOKEN_BATCH_ABI),),
        )
        return AdaptedModel(
            family=self.name, architecture=architecture,
            dense=tuple(checkpoint.tensors[name] for name in sorted(expected)),
            experts=tuple(routed_sources),
            source_tensor_count=len(checkpoint.tensors),
            runtime_topology=topology,
            dense_float32=frozenset(float32),
            dense_fp4=frozenset(fp4),
            dense_int64=frozenset(int64),
            host_mapped_dense=frozenset(host_mapped),
            auxiliary_dense=frozenset(auxiliary),
            supported_expert_quant_profiles=frozenset(FP4_QUANT_PROFILES),
        )
