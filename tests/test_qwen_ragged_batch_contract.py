"""Model-free production-helper invariants for ragged Qwen AR decode."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.mllm_batch_generator import (
    _absolute_text_position_ids,
    _merge_caches,
    _wrap_batch_caches,
)
from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache
from vmlx_engine.models.qwen4_exp.language import (
    QSAIndexer,
    QSAAttention,
    Qwen4ExpTextArgs,
)


def _args():
    return Qwen4ExpTextArgs(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=1.0,
        mrope_section=[2, 1, 1],
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
    )


def _positions(length, start=0):
    return mx.broadcast_to(mx.arange(start, start + length)[None, None], (3, 1, length))


def _empty_prefix(length):
    cache = MiniMaxM3SparseCache()
    values = mx.zeros((1, 2, length, 8))
    cache.update_and_fetch(values, values)
    cache.update_index(mx.zeros((1, 1, length, 11)))
    return cache


@pytest.mark.parametrize("lengths", [(5, 8), (13, 16), (12, 16), (16, 16)])
def test_ragged_qwen_positions_survive_real_merge_proxy_and_filter(lengths):
    caches = _merge_caches([[_empty_prefix(n)] for n in lengths])
    language = SimpleNamespace(model=SimpleNamespace(fa_idx=0))
    positions = _absolute_text_position_ids(
        mx.zeros((2, 1), dtype=mx.int32), _wrap_batch_caches(caches), language
    )
    assert positions[0, :, 0].tolist() == list(lengths)
    caches[0].filter(mx.array([0]))
    positions = _absolute_text_position_ids(
        mx.zeros((1, 1), dtype=mx.int32), _wrap_batch_caches(caches), language
    )
    assert positions[0, :, 0].tolist() == [lengths[0]]
    caches[0].extend(_merge_caches([[_empty_prefix(7)]])[0])
    positions = _absolute_text_position_ids(
        mx.zeros((2, 1), dtype=mx.int32), _wrap_batch_caches(caches), language
    )
    assert positions[0, :, 0].tolist() == [lengths[0], 7]


@pytest.mark.parametrize("lengths", [(5, 8), (13, 16), (12, 16), (16, 16)])
def test_ragged_qsa_mask_matches_unpadded_rows(lengths):
    mx.random.seed(27)
    indexer = QSAIndexer(_args())
    caches = []
    for length in lengths:
        cache = MiniMaxM3SparseCache()
        kv = mx.zeros((1, 2, length, 8))
        cache.update_and_fetch(kv, kv)
        indexer(mx.random.normal((1, length, 32)), cache, offset=0,
                position_ids=_positions(length))
        caches.append(cache)
    merged = _merge_caches([[cache] for cache in caches])[0]
    wrapped = _wrap_batch_caches([merged])[0]
    current = mx.random.normal((2, 1, 32))
    physical_offset = wrapped.offset
    kv = mx.zeros((2, 2, 1, 8))
    wrapped.update_and_fetch(kv, kv)
    positions = mx.array(lengths)[None, :, None]
    positions = mx.broadcast_to(positions, (3, 2, 1))
    actual = indexer(current, wrapped, offset=physical_offset, position_ids=positions)
    width = max(lengths) + 1
    actual = mx.zeros((2, 1, 1, width)) if actual is None else actual
    for row, (length, cache) in enumerate(zip(lengths, caches)):
        kv = mx.zeros((1, 2, 1, 8))
        cache.update_and_fetch(kv, kv)
        expected = indexer(current[row:row + 1], cache, offset=length,
                           position_ids=_positions(1, length))
        expected = mx.zeros((1, 1, 1, length + 1)) if expected is None else expected
        padding = max(lengths) - length
        assert bool(mx.all(mx.isneginf(actual[row, :, :, :padding])))
        np.testing.assert_array_equal(np.asarray(actual[row:row + 1, :, :, padding:]),
                                      np.asarray(expected))
    assert merged.derived == {}


@pytest.mark.parametrize("lengths", [(5, 8), (13, 16), (12, 16), (16, 16)])
def test_ragged_qsa_attention_matches_unpadded_rows(lengths):
    mx.random.seed(28)
    attention = QSAAttention(_args())
    attention.eval()
    caches = []
    for length in lengths:
        cache = MiniMaxM3SparseCache()
        attention(mx.random.normal((1, length, 32)), cache=cache,
                  position_ids=_positions(length))
        caches.append(cache)
    merged = _merge_caches([[cache] for cache in caches])[0]
    current = mx.random.normal((2, 1, 32))
    positions = mx.broadcast_to(mx.array(lengths)[None, :, None], (3, 2, 1))
    actual = attention(current, cache=_wrap_batch_caches([merged])[0],
                       position_ids=positions)
    expected = mx.concatenate([
        attention(current[row:row + 1], cache=cache,
                  position_ids=_positions(1, length))
        for row, (length, cache) in enumerate(zip(lengths, caches))
    ])
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_multitoken_qsa_keeps_prior_batch_dispatch(monkeypatch):
    """The ragged AR correction does not introduce a batched-prefill policy."""
    from vmlx_engine.models.minimax_m3.cache import BatchMiniMaxM3SparseCache

    mx.random.seed(29)
    indexer = QSAIndexer(_args())
    cache = BatchMiniMaxM3SparseCache([0, 2])
    wrapped = _wrap_batch_caches([cache])[0]
    current = mx.random.normal((2, 3, 32))
    kv = mx.zeros((2, 2, 3, 8))
    cache.update_and_fetch(kv, kv)
    calls = []
    original = indexer._mask_from_payload

    def record_dispatch(q, payload, **kwargs):
        calls.append((tuple(q.shape[:2]), kwargs["cache"], kwargs["offset"]))
        return original(q, payload, **kwargs)

    monkeypatch.setattr(indexer, "_mask_from_payload", record_dispatch)
    mask = indexer(current, wrapped, offset=0,
                   position_ids=mx.broadcast_to(_positions(3), (3, 2, 3)))
    assert mask is None
    assert calls == [((2, 3), wrapped, 0)]
