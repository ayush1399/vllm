# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import tokenizers.decoders
from tokenizers import AddedToken, Tokenizer, decoders, models
from tokenizers.tokenizers import decoders as native_decoders
from transformers import AutoTokenizer, TokenizersBackend

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import (
    INVALID_PREFIX_ERR_MSG,
    IncrementalDetokenizer,
    _get_decode_stream_prompt_ids,
)

# ruff: noqa: E501


@pytest.fixture
def byte_fallback_tokenizer():
    pieces = ["<unk>", "<pad>", "<s>", "</s>", "a", "b", "▁x", " X", "é", "�", ""]
    pieces += [f"<0x{byte:02X}>" for byte in range(256)]
    pieces += ["<0x+F>", "<0x+0>", "<0xGG>"]
    native = Tokenizer(
        models.BPE({piece: i for i, piece in enumerate(pieces)}, [], unk_token="<unk>")
    )
    native.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
    )
    native.add_special_tokens(
        [AddedToken(piece, special=True) for piece in ("<pad>", "<s>", "</s>")]
    )
    return TokenizersBackend(
        tokenizer_object=native,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        clean_up_tokenization_spaces=False,
    )


def _request(prompt, skip_special_tokens, spaces_between_special_tokens=True):
    return EngineCoreRequest(
        request_id="bounded-prompt",
        external_req_id="bounded-prompt-external",
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=SamplingParams(
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        ),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _full_prefix_deltas(tokenizer, prompt, generated, skip, spaces=True):
    native = tokenizer._tokenizer
    stream = native_decoders.DecodeStream(ids=prompt, skip_special_tokens=skip)
    added = native.get_added_tokens_decoder()
    last_special = False
    deltas = []
    for token in generated:
        try:
            text = stream.step(native, token)
        except Exception as exc:
            if not str(exc).startswith(INVALID_PREFIX_ERR_MSG):
                raise
            stream = native_decoders.DecodeStream(skip_special_tokens=skip)
            text = stream.step(native, token)
        if not skip and not spaces:
            special = added.get(token)
            if special is not None and last_special:
                text = special.content
            last_special = special is not None
        deltas.append(text or "")
    return deltas


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("spaces", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "special_boundary",
        "pending_bytes",
        "malformed_run",
        "signed_hex",
        "empty",
        "replacement",
        "invalid_prefix",
    ],
)
def test_fast_detokenizer_bounded_prompt_preserves_stream(
    byte_fallback_tokenizer, case, skip, spaces
):
    """Prompt priming preserves native full-prefix text, deltas, and token counts."""
    tokenizer = byte_fallback_tokenizer
    token = tokenizer._tokenizer.token_to_id
    prefix = [token("a")] * 64
    generated = [
        token(" X"),
        token("<s>"),
        token("</s>"),
        token("<0xE2>"),
        token("<0x82>"),
        token("<0xAC>"),
        token("b"),
    ]
    prompts = {
        "ordinary": prefix + [token("▁x")] * 7,
        "special_boundary": [token("<0xE2>"), token("<0x82>")]
        + [token("<pad>")] * 64
        + [token("a")]
        + [token("<pad>")] * 5,
        "pending_bytes": [token("<0xE2>"), token("<0x82>")] + [token("<pad>")] * 64,
        "malformed_run": prefix
        + [token("<0x80>")]
        + [token("<0x61>")] * 100
        + [token("<0xE2>"), token("<0x82>"), token("<0xAC>")],
        "signed_hex": prefix + [token("<0x+F>"), token("<0x+0>")],
        "empty": prefix + [token("")] * 7,
        "replacement": prefix + [token("�")] * 7,
        "invalid_prefix": prefix + [token("<0xF0>")],
    }
    prompt = prompts[case]
    if case == "pending_bytes":
        generated = [token("<0xAC>"), token(" X")]
    elif case == "invalid_prefix":
        generated = [token("<0x9F>"), token("<0x98>"), token("<0x80>"), token(" X")]
    original_prompt = prompt.copy()
    expected = _full_prefix_deltas(tokenizer, prompt, generated, skip, spaces)
    request = _request(prompt, skip, spaces)
    detokenizer = IncrementalDetokenizer.from_new_request(tokenizer, request)
    for i, token_id in enumerate(generated):
        detokenizer.update([token_id], False)
        assert (
            detokenizer.get_next_output_text(i == len(generated) - 1, delta=True)
            == expected[i]
        )
        assert detokenizer.get_next_output_text(False, delta=False) == "".join(
            expected[: i + 1]
        )
        assert detokenizer.output_token_ids == generated[: i + 1]
        assert detokenizer.num_output_tokens() == i + 1
        assert request.prompt_token_ids is prompt
        assert prompt == original_prompt


@pytest.mark.parametrize("case", ["pending_bytes", "malformed_run"])
def test_unguarded_prompt_window_changes_native_output(byte_fallback_tokenizer, case):
    """A raw slice without the eligibility guards loses bytes or leaks prompt text."""
    tokenizer = byte_fallback_tokenizer
    token = tokenizer._tokenizer.token_to_id
    if case == "pending_bytes":
        prompt = [token("<0xE2>"), token("<0x82>")] + [token("<pad>")] * 64
        generated = [token("<0xAC>")]
    else:
        prompt = (
            [token("a")] * 64
            + [token("<0x80>")]
            + [token("<0x61>")] * 100
            + [token("<0xE2>"), token("<0x82>"), token("<0xAC>")]
        )
        generated = [token(" X")]
    full = _full_prefix_deltas(tokenizer, prompt, generated, True)
    naive = _full_prefix_deltas(tokenizer, prompt[-7:], generated, True)
    assert full != naive
    assert _get_decode_stream_prompt_ids(tokenizer._tokenizer, prompt, True) is prompt


@pytest.mark.parametrize("tail", [["a"] * 7, ["<pad>", "a"] + ["<pad>"] * 5])
def test_native_visible_prompt_tail_bounds_priming(byte_fallback_tokenizer, tail):
    tokenizer = byte_fallback_tokenizer._tokenizer
    prompt = [tokenizer.token_to_id("b")] * 64 + [
        tokenizer.token_to_id(piece) for piece in tail
    ]
    original_prompt = prompt.copy()
    assert _get_decode_stream_prompt_ids(tokenizer, prompt, True) == prompt[-7:]
    assert prompt == original_prompt


@pytest.mark.parametrize("piece", ["<0x+F>", "<0x+0>", "<0xGG>", "", "�", None])
def test_ambiguous_native_prompt_tail_keeps_full_context(
    byte_fallback_tokenizer, piece
):
    tokenizer = byte_fallback_tokenizer._tokenizer
    prompt = [tokenizer.token_to_id("a")] * 64 + [
        100000 if piece is None else tokenizer.token_to_id(piece)
    ] * 7
    assert _get_decode_stream_prompt_ids(tokenizer, prompt, True) is prompt


def test_other_decoder_and_external_stream_keep_full_context(
    byte_fallback_tokenizer, monkeypatch
):
    tokenizer = byte_fallback_tokenizer._tokenizer
    prompt = [tokenizer.token_to_id("a")] * 64
    tokenizer.decoder = decoders.Metaspace()
    assert _get_decode_stream_prompt_ids(tokenizer, prompt, True) is prompt
    tokenizer.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
    )
    native_factory = native_decoders.DecodeStream
    received = []

    def external_stream(*args, **kwargs):
        received.append(kwargs.get("ids"))
        return native_factory(*args, **kwargs)

    monkeypatch.setattr(tokenizers.decoders, "DecodeStream", external_stream)
    request = _request(prompt, True)
    detokenizer = IncrementalDetokenizer.from_new_request(
        byte_fallback_tokenizer, request
    )
    assert received == [prompt]
    assert received[0] is prompt
    detokenizer.update([tokenizer.token_to_id("b")], False)
    assert detokenizer.get_next_output_text(True, delta=True) == "b"


def test_custom_native_decoder_keeps_full_prompt_context(byte_fallback_tokenizer):
    """A working Python decoder need not support fingerprint serialization."""

    class Decoder:
        def decode_chain(self, tokens):
            return tokens

    tokenizer = byte_fallback_tokenizer
    native = tokenizer._tokenizer
    native.decoder = decoders.Decoder.custom(Decoder())
    prompt = [native.token_to_id("a")] * 64
    generated = [native.token_to_id("b")]
    expected = _full_prefix_deltas(tokenizer, prompt, generated, True)
    assert expected == ["b"]
    request = _request(prompt, True)
    detokenizer = IncrementalDetokenizer.from_new_request(tokenizer, request)
    detokenizer.update(generated, False)
    assert detokenizer.get_next_output_text(True, delta=True) == expected[0]
    assert detokenizer.output_token_ids == generated
    assert request.prompt_token_ids is prompt and len(prompt) == 64


def test_fast_inc_detok_invalid_utf8_err_case():
    """Test edge case where tokenizer can produce non-monotonic,
    invalid UTF-8 output, which breaks the internal state of
    tokenizers' DecodeStream.
    See https://github.com/vllm-project/vllm/issues/17448.

    Thanks to reproducer from @fpaupier:
    https://gist.github.com/fpaupier/0ed1375bd7633c5be6c894b1c7ac1be3.
    """
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

    # Create a test request
    prompt_token_ids = [107, 4606, 236787, 107]
    params = SamplingParams(skip_special_tokens=True)
    request = EngineCoreRequest(
        request_id="test",
        external_req_id="test-ext",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    detokenizer = IncrementalDetokenizer.from_new_request(tokenizer, request)

    assert detokenizer.__class__.__name__ == "FastIncrementalDetokenizer", (
        "Should use FastIncrementalDetokenizer by default"
    )

    # Process tokens incrementally
    test_tokens = [
        236840,
        107,
        138,
        236782,
        107,
        140,
        236775,
        6265,
        1083,
        623,
        121908,
        147418,
        827,
        107,
        140,
        236775,
        6265,
        236779,
        2084,
        1083,
        623,
        203292,
        827,
        107,
        140,
        236775,
        6265,
        236779,
        7777,
        1083,
        623,
        121908,
        147418,
        569,
        537,
        236789,
        65880,
        569,
        537,
        236789,
        62580,
        853,
        115693,
        210118,
        35178,
        16055,
        1270,
        759,
        215817,
        4758,
        1925,
        1117,
        827,
        107,
        140,
        236775,
        5654,
        1083,
        623,
        110733,
        46291,
        827,
        107,
        140,
        236775,
        5654,
        236779,
        2084,
        1083,
        623,
        136955,
        56731,
        827,
        107,
        140,
        236775,
        5654,
        236779,
        7777,
        1083,
        623,
        194776,
        2947,
        496,
        109811,
        1608,
        890,
        215817,
        4758,
        1925,
        1117,
        2789,
        432,
        398,
        602,
        31118,
        569,
        124866,
        134772,
        509,
        19478,
        1640,
        33779,
        236743,
        236770,
        236819,
        236825,
        236771,
        432,
        398,
        432,
        237167,
        827,
        107,
        140,
        236775,
        77984,
        1083,
        623,
        2709,
        236745,
        2555,
        513,
        236789,
        602,
        31118,
        569,
    ]

    output = ""
    for i, token_id in enumerate(test_tokens):
        detokenizer.update([token_id], False)

        finished = i == len(test_tokens) - 1
        output += detokenizer.get_next_output_text(finished, delta=True)

    assert (
        output
        == r"""[
  {
    "source": "Résultats",
    "source_type": "CONCEPT",
    "source_description": "Résultats de l'analyse de l'impact des opérations israéliennes sur la frontière libanaise",
    "target": "Israël",
    "target_type": "ORGANIZATION",
    "target_description": "Pays qui a obtenu à sa frontière libanaise « un niveau de calme inédit depuis les années 1960 »",
    "relationship": "Obtention d'un niveau de"""
    )
