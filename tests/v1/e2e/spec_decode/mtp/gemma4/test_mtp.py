# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gzip
import json
import os
from decimal import Decimal

import pytest
import regex as re

from tests.utils import multi_gpu_test, single_gpu_only
from vllm import SamplingParams
from vllm.platforms import current_platform

from .._correctness import check_mtp_correctness


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (
            ("mtp", "google/gemma-4-E4B-it", 1, "google/gemma-4-E4B-it-assistant"),
            False,
            0.50,
        ),  # gemma4 MTP with assistant model, ref: ~62%
    ],
    ids=["gemma4-e4b"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
    )


def _target_full_chain_enabled(model):
    import torch

    from vllm.model_executor.models.gemma4 import (
        Gemma4Model,
    )
    from vllm.models.gemma4.nvidia.specialized_decode.model import (
        _gemma4_can_run_full_chain,
    )

    target = next(m for m in model.modules() if isinstance(m, Gemma4Model))
    x = target.embed_tokens.weight.new_empty((5, 5376))
    assert x.dtype == torch.bfloat16
    return getattr(
        target, "_full_chain_config_supported", False
    ) and _gemma4_can_run_full_chain(target, x, None)


def _has_fp8_assistant(worker):
    from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
    )

    runner = worker.model_runner
    drafter = getattr(runner, "speculator", None) or runner.drafter
    return get_input_quant_key(drafter.model.model.layers[0].self_attn.q_proj) == (
        kFp8Dynamic128Sym
    )


def _target_pdl_enabled(model):
    from vllm.model_executor.models.gemma4 import Gemma4Model

    target = next(m for m in model.modules() if isinstance(m, Gemma4Model))
    return all(
        getattr(layer.mlp.gate_up_proj, "_gemma4_pdl", False) for layer in target.layers
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="SM90 FP8 producers"
)
@pytest.mark.parametrize("num_speculative_tokens", [1, 2, 3, 4])
@multi_gpu_test(num_gpus=8)
def test_fp8_mtp_producer_outputs(
    monkeypatch, vllm_runner, num_speculative_tokens, enable_pickle, tmp_path
):
    """Select the fused target path and answer short arithmetic prompts correctly."""
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_E8M0", "1")
    prompts = [
        "What is 7 times 8? Answer with only the integer.",
        "What is 12 plus 15? Answer with only the integer.",
    ]
    outputs = []
    profile_dir = tmp_path / "profile"
    for enabled in (True, False):
        with vllm_runner(
            os.environ.get("GEMMA4_FP8_MODEL", "RedHatAI/gemma-4-31B-it-FP8-block"),
            additional_config={
                "decode_backend": "latency_specialized" if enabled else "auto"
            },
            tensor_parallel_size=8,
            profiler_config={
                "profiler": "torch",
                "torch_profiler_dir": str(profile_dir),
                "torch_profiler_activities": ["CUDA"],
                "torch_profiler_record_shapes": False,
                "torch_profiler_with_stack": False,
            },
            dtype="bfloat16",
            kv_cache_dtype="bfloat16",
            max_model_len=2048,
            max_num_seqs=4,
            max_num_batched_tokens=2048,
            language_model_only=True,
            hf_overrides={"text_config": {"use_bidirectional_attention": None}},
            attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 4},
            compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 20]},
            speculative_config={
                "method": "mtp",
                "model": os.environ.get(
                    "GEMMA4_MTP_MODEL", "google/gemma-4-31B-it-assistant"
                ),
                "num_speculative_tokens": num_speculative_tokens,
            },
        ) as runner:
            assert runner.llm.apply_model(_target_full_chain_enabled) == [enabled] * 8
            assert runner.llm.apply_model(_target_pdl_enabled) == [enabled] * 8
            fp8_draft = runner.llm.collective_rpc(_has_fp8_assistant)
            if enabled:
                runner.llm.start_profile("mtp-producers")
            generated = []
            for batch in (prompts[:1], prompts):
                generated.extend(
                    runner.llm.chat(
                        [[{"role": "user", "content": prompt}] for prompt in batch],
                        SamplingParams(temperature=0, max_tokens=16),
                        chat_template_kwargs={"enable_thinking": False},
                        use_tqdm=False,
                    )
                )
            outputs.append([output.outputs[0].text for output in generated])
            if enabled:
                runner.llm.stop_profile()
                traces = list(profile_dir.glob("*rank0*.pt.trace.json.gz"))
                assert traces
                names = {
                    event["name"]
                    for path in traces
                    for event in json.loads(gzip.decompress(path.read_bytes()))[
                        "traceEvents"
                    ]
                    if event.get("cat") == "kernel"
                }
                assert any("chain<" in name for name in names)
                assert any("gemma4_pdl_gemm_kernel" in name for name in names)
                assert any("combine_quant" in name for name in names)
                assert not any(
                    "reshape_and_cache_flash_kernel" in name for name in names
                )
                if all(fp8_draft):
                    assert any("_fp8_gemv_kernel" in name for name in names)
                    assert any("_concat_fp8_gemm_kernel" in name for name in names)
    for generated in outputs:
        assert len(generated) == 3
        for text, expected in zip(generated, (56, 56, 27)):
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
            assert numbers, text
            assert Decimal(numbers[-1]) == expected, text
