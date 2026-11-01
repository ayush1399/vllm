# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection and deployment requirements for Gemma4 specialized execution."""

from vllm.config import VllmConfig, get_current_vllm_config_or_none


def pdl_enabled() -> bool:
    config = get_current_vllm_config_or_none()
    if config is None or not specialized_decode_enabled(config):
        return False
    extra = config.additional_config
    if not isinstance(extra, dict):
        return False
    enabled = extra.get("gemma4_pdl", True)
    if not isinstance(enabled, bool):
        raise ValueError("gemma4_pdl must be a boolean")
    return enabled


def specialized_decode_enabled(
    vllm_config: VllmConfig | None = None, *, validate: bool = False
) -> bool:
    if vllm_config is None:
        vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or vllm_config.model_config is None:
        return False
    config = vllm_config.model_config.hf_config
    if not config.model_type.startswith("gemma4"):
        return False
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend == "auto":
        return False
    if backend != "latency_specialized":
        raise ValueError("Gemma4 decode_backend must be auto or latency_specialized")
    if validate:
        reason = unsupported(vllm_config)
        if reason is not None:
            raise ValueError(f"Gemma4 specialized decode does not support {reason}")
    return True


def unsupported(vllm_config: VllmConfig) -> str | None:
    import torch

    from vllm import envs
    from vllm.platforms import current_platform
    from vllm.utils.flashinfer import is_flashinfer_fp8_blockscale_gemm_supported

    from .full_chain import native_available

    model = vllm_config.model_config
    text = model.hf_text_config
    parallel = vllm_config.parallel_config
    spec = vllm_config.speculative_config
    checks = (
        (not current_platform.is_device_capability(90), "GPUs other than SM90"),
        (parallel.tensor_parallel_size != 8, "tensor parallel sizes other than 8"),
        (parallel.pipeline_parallel_size != 1, "pipeline parallelism"),
        (vllm_config.lora_config is not None, "LoRA"),
        (envs.VLLM_BATCH_INVARIANT, "batch invariant execution"),
        (model.dtype != torch.bfloat16, "activation dtypes other than bfloat16"),
        (vllm_config.quant_config is None, "unquantized target weights"),
        (not envs.VLLM_USE_DEEP_GEMM_E8M0, "DeepGEMM without UE8M0 scales"),
        (not envs.VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER, "disabled FlashInfer GEMMs"),
        (
            not is_flashinfer_fp8_blockscale_gemm_supported(),
            "unavailable FlashInfer blockscale GEMMs",
        ),
        (not native_available(), "a build without the Gemma4 CUDA producer"),
        (
            text.hidden_size != 5376 or text.num_hidden_layers != 60,
            "targets other than Gemma4 31B",
        ),
        (text.rms_norm_eps != 1e-6, "RMSNorm epsilon other than 1e-6"),
        (getattr(text, "num_experts", None) is not None, "MoE"),
        (getattr(text, "use_double_wide_mlp", False), "double-width MLPs"),
        (bool(getattr(text, "num_kv_shared_layers", 0)), "target KV-shared layers"),
        (bool(getattr(text, "hidden_size_per_layer_input", 0)), "per-layer embeddings"),
        (vllm_config.cache_config.kv_sharing_fast_prefill, "KV-sharing fast prefill"),
        (
            spec is not None and not spec.use_gemma4_mtp(),
            "speculation methods other than Gemma4 MTP",
        ),
    )
    return next((reason for failed, reason in checks if failed), None)
