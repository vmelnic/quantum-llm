"""Strict source adapter for the upstream Muse Glimmer checkpoint ABI."""

from __future__ import annotations

from .adapters import (
    HIDDEN_BATCH_ABI,
    POSITION_BATCH_ABI,
    TOKEN_BATCH_ABI,
    AdaptedModel,
    RuntimeLayerTopology,
    RuntimeModelTopology,
    RuntimeOperationTopology,
    _float32_bits,
    _integer,
    _number,
)
from .constants import FP4_QUANT_PROFILE
from .errors import AdapterError
from .safetensors import SafeTensorCheckpoint


class MuseGlimmerAdapter:
    """Map Muse source names to generic dense-FP4 operation ABI 2."""

    name = "muse_glimmer"

    def adapt(self, checkpoint: SafeTensorCheckpoint) -> AdaptedModel:
        config = checkpoint.config
        if config.get("model_type") != "muse_glimmer":
            raise AdapterError("muse_glimmer adapter requires model_type muse_glimmer")
        architectures = config.get("architectures")
        if not isinstance(architectures, list) or architectures != [
            "MuseGlimmerForConditionalGeneration"
        ]:
            raise AdapterError("muse_glimmer architecture is unsupported")
        text = config.get("text_config")
        vision = config.get("vision_config")
        if not isinstance(text, dict) or not isinstance(vision, dict):
            raise AdapterError("muse_glimmer text/vision configs are absent")
        if text.get("model_type") != "muse_glimmer_text":
            raise AdapterError("muse_glimmer text_config type is unsupported")
        if vision.get("model_type") != "muse_glimmer_vision":
            raise AdapterError("muse_glimmer vision_config type is unsupported")

        layers = _integer(text, "num_hidden_layers")
        hidden = _integer(text, "hidden_size")
        intermediate = _integer(text, "intermediate_size")
        vocab = _integer(text, "vocab_size")
        max_context = _integer(text, "max_position_embeddings")
        heads = _integer(text, "num_attention_heads")
        kv_heads = _integer(text, "num_key_value_heads")
        head_dim = _integer(text, "head_dim")
        window = _integer(text, "sliding_window")
        epsilon = _number(text, "rms_norm_eps")
        post_epsilon = _number(text, "post_norm_eps")
        query_scale = _number(text, "qk_scale_factor")
        output_multiplier = _number(text, "output_multiplier")
        logit_softcap = _number(text, "final_logit_softcapping")
        if (
            hidden <= 0
            or heads * head_dim <= 0
            or heads % kv_heads
            or window > max_context
            or text.get("hidden_activation") != "silu"
            or bool(text.get("attention_bias", True))
            or bool(text.get("tie_word_embeddings", True))
        ):
            raise AdapterError("muse_glimmer text geometry is inconsistent")
        layer_types = text.get("layer_types")
        layer_thetas = text.get("layer_rope_theta")
        if (
            not isinstance(layer_types, list)
            or len(layer_types) != layers
            or any(item not in {"sliding_attention", "full_attention"}
                   for item in layer_types)
            or not isinstance(layer_thetas, list)
            or len(layer_thetas) != layers
        ):
            raise AdapterError("muse_glimmer layer topology is inconsistent")
        for layer_type, theta in zip(layer_types, layer_thetas, strict=True):
            if isinstance(theta, bool) or not isinstance(theta, (int, float)):
                raise AdapterError("muse_glimmer layer RoPE value is invalid")
            if (layer_type == "sliding_attention") != (float(theta) > 0.0):
                raise AdapterError("muse_glimmer RoPE/window topology disagrees")

        vision_layers = _integer(vision, "num_hidden_layers")
        vision_hidden = _integer(vision, "hidden_size")
        vision_intermediate = _integer(vision, "intermediate_size")
        vision_heads = _integer(vision, "num_attention_heads")
        vision_positions = _integer(vision, "max_position_embeddings")
        patch = _integer(vision, "patch_size")
        temporal_patch = _integer(vision, "patch_temporal")
        merge = _integer(vision, "merge_size")
        projector_hidden = _integer(config, "projector_hidden_size")
        merged_vision = vision_hidden * merge * merge
        if (
            vision_hidden % vision_heads
            or merged_vision != _integer(config, "out_hidden_size")
            or vision.get("hidden_act") != "gelu"
            or config.get("projector_hidden_act") != "gelu"
        ):
            raise AdapterError("muse_glimmer vision geometry is inconsistent")
        vision_layer_types = vision.get("layer_types")
        if (
            not isinstance(vision_layer_types, list)
            or len(vision_layer_types) != vision_layers
            or any(item not in {"window_attention", "full_attention"}
                   for item in vision_layer_types)
        ):
            raise AdapterError("muse_glimmer vision topology is inconsistent")

        expected: dict[str, tuple[int, ...]] = {}
        fp4: set[str] = set()
        float32: set[str] = set()
        auxiliary: set[str] = set()

        def add(
            name: str,
            shape: tuple[int, ...],
            *,
            matrix: bool = False,
            aux: bool = False,
        ) -> None:
            if name in expected:
                raise AdapterError(f"duplicate muse_glimmer tensor {name}")
            expected[name] = shape
            (fp4 if matrix else float32).add(name)
            if aux:
                auxiliary.add(name)

        text_prefix = "model.language_model."
        add(text_prefix + "embed_tokens.weight", (vocab, hidden), matrix=True)
        add(text_prefix + "norm.weight", (hidden,))
        add("lm_head.weight", (vocab, hidden), matrix=True)
        query_width = heads * head_dim
        kv_width = kv_heads * head_dim
        for layer in range(layers):
            prefix = f"{text_prefix}layers.{layer}."
            for name in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ):
                add(prefix + name, (hidden,))
            add(prefix + "self_attn.q_proj.weight", (query_width, hidden), matrix=True)
            add(prefix + "self_attn.gate_proj.weight", (query_width, hidden), matrix=True)
            add(prefix + "self_attn.k_proj.weight", (kv_width, hidden), matrix=True)
            add(prefix + "self_attn.v_proj.weight", (kv_width, hidden), matrix=True)
            add(prefix + "self_attn.o_proj.weight", (hidden, query_width), matrix=True)
            add(prefix + "mlp.gate_proj.weight", (intermediate, hidden), matrix=True)
            add(prefix + "mlp.up_proj.weight", (intermediate, hidden), matrix=True)
            add(prefix + "mlp.down_proj.weight", (hidden, intermediate), matrix=True)

        vision_prefix = "model.vision_tower."
        patch_width = 3 * temporal_patch * patch * patch
        for name, shape, matrix in (
            ("patch_embedder.patch_embedding.weight", (vision_hidden, patch_width), True),
            ("patch_embedder.position_embedding_table.weight", (vision_positions, vision_hidden), True),
            ("ln_pre.weight", (vision_hidden,), False),
            ("ln_pre.bias", (vision_hidden,), False),
            ("ln_post.weight", (vision_hidden,), False),
            ("ln_post.bias", (vision_hidden,), False),
        ):
            add(vision_prefix + name, shape, matrix=matrix, aux=True)
        for layer in range(vision_layers):
            prefix = f"{vision_prefix}layers.{layer}."
            for norm in ("norm1", "norm2"):
                add(prefix + norm + ".weight", (vision_hidden,), aux=True)
                add(prefix + norm + ".bias", (vision_hidden,), aux=True)
            for projection in ("q_proj", "k_proj", "v_proj", "proj"):
                add(prefix + f"attn.{projection}.weight",
                    (vision_hidden, vision_hidden), matrix=True, aux=True)
                add(prefix + f"attn.{projection}.bias", (vision_hidden,), aux=True)
            add(prefix + "mlp.fc1.weight",
                (vision_intermediate, vision_hidden), matrix=True, aux=True)
            add(prefix + "mlp.fc1.bias", (vision_intermediate,), aux=True)
            add(prefix + "mlp.fc2.weight",
                (vision_hidden, vision_intermediate), matrix=True, aux=True)
            add(prefix + "mlp.fc2.bias", (vision_hidden,), aux=True)
        add("model.vision_adapter.fc1.weight",
            (projector_hidden, merged_vision), matrix=True, aux=True)
        add("model.vision_adapter.fc2.weight",
            (projector_hidden, projector_hidden), matrix=True, aux=True)
        add("model.vision_projection.weight",
            (hidden, projector_hidden), matrix=True, aux=True)

        actual = set(checkpoint.tensors)
        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            unknown = sorted(actual - set(expected))
            raise AdapterError(
                "muse_glimmer tensor partition mismatch; "
                f"missing={missing[:8]}, unknown={unknown[:8]}"
            )
        for name, shape in expected.items():
            info = checkpoint.tensors[name]
            if info.shape != shape or info.dtype != "BF16":
                raise AdapterError(
                    f"muse_glimmer tensor mismatch for {name}: "
                    f"expected BF16 {shape}, got {info.dtype} {info.shape}"
                )

        norm_bits = _float32_bits(epsilon)
        post_norm_bits = _float32_bits(post_epsilon)
        query_scale_bits = _float32_bits(query_scale)
        output_multiplier_bits = _float32_bits(output_multiplier)
        softcap_bits = _float32_bits(logit_softcap)
        base_theta = _number(text.get("rope_parameters"), "rope_theta")
        attributes = (
            ("attention_heads", heads),
            ("kv_heads", kv_heads),
            ("head_dim", head_dim),
            ("rotary_dimension", head_dim),
            ("full_attention_layers", layers),
            ("mtp_layers", 0),
            ("norm_epsilon_f32_bits", norm_bits),
            ("rope_theta_f32_bits", _float32_bits(base_theta)),
            ("zero_centered_norm", 0),
        )
        required = (
            ("embedding.lookup.fp4-block32.v1", 2),
            ("block.full-attention.output-gated.v1", 2),
            ("ffn.swiglu.dense.fp4-block32.v1", 2),
            ("head.rmsnorm.token-select.fp4-block32.v1", 2),
        )
        runtime_layers = tuple(
            RuntimeLayerTopology(
                logical_layer=layer,
                block_capability="block.full-attention.output-gated.v1",
                block_abi=2,
                routed_component=None,
                component_layer=0,
                parameters=(("attention_window_tokens",
                             window if layer_types[layer] == "sliding_attention" else 0),),
            )
            for layer in range(layers)
        )
        operations: list[RuntimeOperationTopology] = [
            RuntimeOperationTopology(
                logical_layer=None,
                capability="embedding.lookup.fp4-block32.v1",
                abi=2,
                routed_component=None,
                component_layer=0,
                parameters=(
                    ("output_norm_mode", 1),
                    ("output_norm_epsilon_f32_bits", norm_bits),
                ),
                tensor_bindings=(("weight", text_prefix + "embed_tokens.weight"),),
                input_bindings=(("token_ids", "request.token_ids", TOKEN_BATCH_ABI),),
                output_bindings=(("hidden", "hidden.0", HIDDEN_BATCH_ABI),),
            )
        ]
        for layer, layer_type in enumerate(layer_types):
            prefix = f"{text_prefix}layers.{layer}."
            after_attention = f"layer.{layer}.after_attention"
            theta = float(layer_thetas[layer])
            attention_parameters = [
                ("input_norm_mode", 1),
                ("input_norm_epsilon_f32_bits", norm_bits),
                ("qk_norm_mode", 1),
                ("qk_norm_epsilon_f32_bits", norm_bits),
                ("query_scale_f32_bits", query_scale_bits),
                ("attention_window_tokens", window if layer_type == "sliding_attention" else 0),
                ("rotary_enabled", 1 if theta > 0.0 else 0),
                ("post_norm_mode", 1),
                ("post_norm_epsilon_f32_bits", post_norm_bits),
            ]
            if theta > 0.0:
                attention_parameters.append(("rope_theta_f32_bits", _float32_bits(theta)))
            operations.append(RuntimeOperationTopology(
                logical_layer=layer,
                capability="block.full-attention.output-gated.v1",
                abi=2,
                routed_component=None,
                component_layer=0,
                parameters=tuple(attention_parameters),
                tensor_bindings=(
                    ("input_norm", prefix + "input_layernorm.weight"),
                    ("query_projection", prefix + "self_attn.q_proj.weight"),
                    ("gate_projection", prefix + "self_attn.gate_proj.weight"),
                    ("key_projection", prefix + "self_attn.k_proj.weight"),
                    ("value_projection", prefix + "self_attn.v_proj.weight"),
                    ("output_projection", prefix + "self_attn.o_proj.weight"),
                    ("post_norm", prefix + "post_attention_layernorm.weight"),
                ),
                input_bindings=(
                    ("hidden", f"hidden.{layer}", HIDDEN_BATCH_ABI),
                    ("positions", "request.positions", POSITION_BATCH_ABI),
                ),
                output_bindings=(("hidden", after_attention, HIDDEN_BATCH_ABI),),
            ))
            operations.append(RuntimeOperationTopology(
                logical_layer=layer,
                capability="ffn.swiglu.dense.fp4-block32.v1",
                abi=2,
                routed_component=None,
                component_layer=0,
                parameters=(
                    ("input_norm_mode", 1),
                    ("input_norm_epsilon_f32_bits", norm_bits),
                    ("post_norm_mode", 1),
                    ("post_norm_epsilon_f32_bits", post_norm_bits),
                ),
                tensor_bindings=(
                    ("input_norm", prefix + "pre_feedforward_layernorm.weight"),
                    ("gate_projection", prefix + "mlp.gate_proj.weight"),
                    ("up_projection", prefix + "mlp.up_proj.weight"),
                    ("down_projection", prefix + "mlp.down_proj.weight"),
                    ("post_norm", prefix + "post_feedforward_layernorm.weight"),
                ),
                input_bindings=(("hidden", after_attention, HIDDEN_BATCH_ABI),),
                output_bindings=(("hidden", f"hidden.{layer + 1}", HIDDEN_BATCH_ABI),),
            ))
        operations.append(RuntimeOperationTopology(
            logical_layer=None,
            capability="head.rmsnorm.token-select.fp4-block32.v1",
            abi=2,
            routed_component=None,
            component_layer=0,
            parameters=(
                ("logit_multiplier_f32_bits", output_multiplier_bits),
                ("logit_softcap_f32_bits", softcap_bits),
            ),
            tensor_bindings=(
                ("norm", text_prefix + "norm.weight"),
                ("weight", "lm_head.weight"),
            ),
            input_bindings=(("hidden", f"hidden.{layers}", HIDDEN_BATCH_ABI),),
            output_bindings=(("token_ids", "response.token_ids", TOKEN_BATCH_ABI),),
        ))
        architecture = {
            "family": self.name,
            "model_type": self.name,
            "upstream_model_type": "muse_glimmer",
            "architectures": architectures,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "vocab_size": vocab,
            "max_position_embeddings": max_context,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "hidden_activation": "silu",
            "sliding_window": window,
            "sliding_attention_layers": layer_types.count("sliding_attention"),
            "full_attention_layers": layer_types.count("full_attention"),
            "rms_norm_epsilon": epsilon,
            "post_norm_epsilon": post_epsilon,
            "qk_scale_factor": query_scale,
            "output_multiplier": output_multiplier,
            "final_logit_softcapping": logit_softcap,
            "vision_auxiliary_only": True,
        }
        topology = RuntimeModelTopology(
            architecture_id=self.name,
            vocab_size=vocab,
            max_context_tokens=max_context,
            hidden_size=hidden,
            attributes=attributes,
            required_kernels=required,
            components=(),
            layers=runtime_layers,
            operations=tuple(operations),
            tensor_bindings=(
                ("token_embedding", text_prefix + "embed_tokens.weight"),
                ("final_norm", text_prefix + "norm.weight"),
                ("output_head", "lm_head.weight"),
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
            dense=tuple(checkpoint.tensors[name] for name in sorted(expected)),
            experts=(),
            source_tensor_count=len(checkpoint.tensors),
            runtime_topology=topology,
            dense_float32=frozenset(float32),
            dense_fp4=frozenset(fp4),
            auxiliary_dense=frozenset(auxiliary),
            supported_expert_quant_profiles=frozenset((FP4_QUANT_PROFILE,)),
        )
