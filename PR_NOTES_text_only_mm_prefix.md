# WIP notes: text-only Gemma4ForCausalLM keeps mm_prefix on

> Handoff notes for continuing this work. **Drop this commit before opening the PR**
> (`git rebase -i` and drop "WIP: handoff notes", or `git reset --hard HEAD~1` if it
> is still the tip).

Branch: `fix-text-only-mm-prefix`, based on upstream main `b4b7eca0ef` (2026-10-14).

## Problem

`ModelConfig._supports_multimodal_for_mm_prefix` treated `multimodal_config is None`
as "early call before multimodal init" and returned `True`. A **non-multimodal
architecture never builds a multimodal config**, so it stayed `True` forever, and the
arch config was only rebuilt inside the `if self._model_info.supports_multimodal:` block.

Effect: a `Gemma4ForCausalLM` checkpoint whose text config carries
`use_bidirectional_attention: "vision"` (inherited from the multimodal parent) gets
`is_mm_prefix_lm=True` via `Gemma4ModelArchConfigConvertor`, even though it can never
see image tokens. That changes attention backend selection:

- SM90: `TRITON_FLASH_ATTN` composite is put first instead of plain `FLASH_ATTN`
  (`vllm/platforms/cuda.py` ~L170).
- SM100: `TRITON_FLASHINFER` composite instead of `FLASHINFER`.
- Backends without `supports_mm_prefix()` are vetoed.

PR #48796 (merged, fe5145765f) fixed the *configuration-disabled* case (multimodal
checkpoint served with all mm limits = 0 / `--language-model-only`), but not this one.

## Fix (commit 1)

`vllm/config/model.py`:

1. `_supports_multimodal_for_mm_prefix`: when `multimodal_config is None`, return
   `True` only before `_model_info` is resolved; afterwards return
   `model_info.supports_multimodal` (False for text-only archs).
   Note: the first `get_model_arch_config()` call (~L683) runs **before**
   `self._model_info` is set (~L731), hence the `getattr(..., None)`.
2. `__post_init__`: `elif self.model_arch_config.is_mm_prefix_lm:` rebuild the arch
   config for text-only archs. Gated so other text-only models don't pay for an
   extra convert (which can hit `get_quantization_config`).

Unchanged paths, checked by reading the code:

- Multimodal models: `multimodal_config` is non-None after init → same cached path.
- `with_hf_config` copies (`vllm/config/vllm.py` ~L988): deep-copy keeps a non-None
  `multimodal_config` and the sticky cache → unchanged.
- Drafters: `get_model(vllm_config=draft_vllm_config, model_config=draft_model_config)`;
  `Attention` reads `vllm_config.model_config` (target), so drafter attention is
  unaffected. DFlash already clears mm_prefix explicitly (`dflash.py` ~L96).
  `SpeculativeConfig.update_arch_` calls `get_model_arch_config()` with a stale
  `_model_info`, but only for draft configs whose attention doesn't read it.

Tests (`tests/config/test_multimodal_config.py`):

- Renamed/updated `test_supports_multimodal_for_mm_prefix_before_multimodal_config`
  → `test_supports_multimodal_for_mm_prefix_text_only_model` (asserted the old buggy
  `True` for TinyLlama; now `False`, still no sticky-cache write).
- Added `test_text_only_gemma4_clears_mm_prefix` (tiny `Gemma4TextConfig` saved to
  `tmp_path`, no weights).

## Duplicate check (done 2026-10-14 via GitHub search API; `gh` not installed there)

No open PR/issue for this. Related but different:

- #48796 (merged): config-disabled text-only serving of mm checkpoints.
- #47118 (open): runtime FlashInfer/Triton switching for multimodal Gemma 4.
- #46558 (open): FlashInfer mm-prefix support (would reduce the veto's impact).

Re-run before opening the PR:

```bash
gh pr list --repo vllm-project/vllm --state open --search "mm_prefix"
gh pr list --repo vllm-project/vllm --state open --search "is_mm_prefix_lm text-only"
gh pr list --repo vllm-project/vllm --state open --search "Gemma4ForCausalLM"
```

## Checkpoints for the serving repro

No official Google checkpoint reproduces it: all `google/gemma-4-*` are
`Gemma4ForConditionalGeneration`. E2B/E4B have `use_bidirectional_attention=None`,
so they don't trigger it either (don't use the registry example `google/gemma-4-E2B-it`).

Community text-only extractions that DO trigger it (`Gemma4ForCausalLM`,
`gemma4_text`, `use_bidirectional_attention: "vision"`):

| Checkpoint | Base | Quant | Notes |
| --- | --- | --- | --- |
| **`bahadirakdemir/gemma-4-31B-it-text-fp8`** | 31B dense | ModelOpt FP8 | **Recommended** (most downloads, FP8 native on H100) |
| `xbill9/gemma-4-31B-it-qat-q4_0-w4a16-ct-text` | 31B dense | CT W4A16 | - |
| `xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text` | 26B-A4B MoE | CT W4A16 | small, fits 1 GPU |
| `sartajbhuvaji/Gemma-4-31B-IT-NVFP4-FP8attn-text` | 31B dense | ModelOpt NVFP4 | Blackwell only |

Not yet verified that these load cleanly in vLLM.

Fallback (official weights): `google/gemma-4-31B-it` with
`--hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}'`. Unverified that
`Gemma4ForCausalLM` loads the mm checkpoint's weights cleanly.

Control (should be unchanged by the fix): `google/gemma-4-31B-it` as-is → still the
mm_prefix composite.

## Results so far

Config-level repro (no GPU), tiny `Gemma4TextConfig(architectures=["Gemma4ForCausalLM"],
use_bidirectional_attention="vision", ...)`:

- main: `is_mm_prefix_lm=True`
- branch: `is_mm_prefix_lm=False`

```bash
.venv/bin/python -m pytest tests/config/test_multimodal_config.py -q
# 84 passed; both new/updated tests FAIL without the fix
.venv/bin/python -m pytest tests/config tests/test_config.py -q \
    --ignore=tests/config/test_speculative_draft_hf_overrides.py
# 672 passed, 40 failed, 5 skipped -- the same 40 fail on unpatched main
# (gated HF models / ImportErrors from transformers 5.14.1; main wants 5.18)
pre-commit run --files vllm/config/model.py tests/config/test_multimodal_config.py
# all passed (ruff, mypy, etc.)
```

Env caveat: tested with an older venv (transformers 5.14.1) and compiled `.so` files
symlinked from another checkout. Re-run in a clean env on the new machine:

```bash
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test/cuda.in
```

(If on a detached/old commit, pin `VLLM_PRECOMPILED_WHEEL_LOCATION` so the
precompiled wheel matches; on a branch off current main nightly is fine.)

## TODO

1. Re-run the unit tests above in a clean env.
2. Serving before/after on H100 (main vs this branch), e.g.:
   ```bash
   vllm serve bahadirakdemir/gemma-4-31B-it-text-fp8 -tp 2 2>&1 | tee serve.log
   grep -iE "attention backend|mm_prefix|heterogeneous head" serve.log
   ```
   Expect main → `TRITON_FLASH_ATTN` composite; branch → `FLASH_ATTN`.
3. `vllm bench serve` TPOT/TTFT at concurrency 1 and 32 for both. Gain on Hopper may
   be small (composite already routes causal to FA); on B200 expect larger
   (FLASHINFER). Measure, don't assume.
4. Accuracy: GSM8K / lm_eval on both builds to show no output change
   (AGENTS.md requires evals for serving-affecting changes).
5. Control: `google/gemma-4-31B-it` still selects the composite on the branch.
6. Drop the notes commit, open the PR.

## PR description draft

**Title:** `[Bugfix] Don't enable mm_prefix attention for text-only architectures`

> `ModelConfig._supports_multimodal_for_mm_prefix` returned `True` whenever
> `multimodal_config` was `None`, intended for the early call before multimodal init.
> Text-only architectures never build a multimodal config, so they kept
> `is_mm_prefix_lm=True` whenever their HF config requested it. This happens in
> practice with text-only Gemma 4 extractions (`Gemma4ForCausalLM`, e.g.
> `bahadirakdemir/gemma-4-31B-it-text-fp8`), which inherit
> `use_bidirectional_attention: "vision"` and therefore get the mm-prefix
> composite backend / veto backends without mm_prefix support, despite never seeing
> image tokens.
>
> Fix: only keep mm_prefix before the architecture is resolved; afterwards a missing
> multimodal config means text-only. Rebuild the arch config for text-only models
> that had mm_prefix on.
>
> Not a duplicate: #48796 covers multimodal checkpoints with mm inputs
> configuration-disabled; #47118 does runtime backend switching for multimodal Gemma 4.
>
> Tests: <commands + results>. Serving: <backend log before/after, bench numbers>.
> Evals: <GSM8K before/after>.
>
> AI assistance (Claude Code) was used; I reviewed every changed line and ran the
> tests above.
