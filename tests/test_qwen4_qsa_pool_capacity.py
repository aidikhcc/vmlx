"""Bounded capacity is derived-only; old graph views and rollback stay exact."""

import mlx.core as mx
import pytest

from vmlx_engine.models.qwen4_exp.language import (
    QSAIndexer, QSACache, Qwen4ExpTextArgs,
    _QSACapacityPooledFrontier, _QSAPooledFrontier,
)


@pytest.mark.parametrize("length", [1, 255, 256, 257, 1023])
def test_append_growth_and_physical_accounting(length):
    fr = _QSACapacityPooledFrontier(4, 1)
    first = mx.arange(length * 16, dtype=mx.float32).reshape(1, length, 16)
    second = -mx.ones((1, 3, 16))
    mx.eval(fr.append(first, 1024 * 1024))
    assert fr.nbytes == fr._buffer.nbytes
    assert fr.nbytes >= fr.pooled.nbytes
    old = fr.pooled
    out = fr.append(second, 1024 * 1024)
    mx.eval(out, old)
    assert mx.array_equal(old, first)
    assert mx.array_equal(out, mx.concatenate([first, second], axis=1))
    assert fr.blocks == length + 3


def test_lazy_old_view_survives_trim_and_overwrite():
    fr = _QSACapacityPooledFrontier(4, 1)
    original = mx.arange(16 * 16, dtype=mx.float32).reshape(1, 16, 16)
    old = fr.append(original, 65536)
    old_result = old * 2  # Deliberately unevaluated old graph.
    fr.truncate_to_tokens(13 * 4 + 2)
    replacement = -mx.ones((1, 5, 16))
    new = fr.append(replacement, 65536)
    mx.eval(old_result, new)
    assert mx.array_equal(old_result, original * 2)
    assert mx.array_equal(new, mx.concatenate([original[:, :13], replacement], axis=1))
    fr.truncate_to_tokens(0)
    assert fr.blocks == 0 and fr.pooled is None and fr.nbytes == 0


def test_capacity_reserve_respects_non_step_aligned_cap():
    fr = _QSACapacityPooledFrontier(4, 1)
    cap = 7 * 16 * 4
    a = mx.ones((1, 5, 16))
    mx.eval(fr.append(a, cap))
    assert fr._buffer.shape == (1, 7, 16)
    assert fr.nbytes == cap
    b = mx.zeros((1, 3, 16))
    out = fr.append(b, cap)
    mx.eval(out)
    assert mx.array_equal(out, mx.concatenate([a, b], axis=1))
    assert fr.evicted == 1 and fr.nbytes == 0 and fr.blocks == 0


def test_zero_budget_returns_exact_without_retention():
    fr = _QSACapacityPooledFrontier(4, 1)
    a = mx.ones((1, 3, 16))
    assert mx.array_equal(fr.append(a, 0), a)
    assert fr.pooled is None and fr.nbytes == 0


def test_default_off_flag_transition_and_live_cap_shrink(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_QSA_POOL_CAPACITY", raising=False)
    monkeypatch.delenv("VMLX_QWEN4_QSA_POOL_RETAIN_MAX_MB", raising=False)
    ix = QSAIndexer(Qwen4ExpTextArgs(
        hidden_size=32, indexer_n_heads=2, indexer_head_dim=16,
        indexer_budget=8, indexer_compress_ratio=4, head_dim=32,
        partial_rotary_factor=0.25, mrope_section=[2, 1, 1],
    ))
    cache = QSACache()
    keys = mx.arange(12 * 16, dtype=mx.float32).reshape(1, 12, 16)
    pos = mx.broadcast_to(mx.arange(12)[None, :, None], (1, 12, 3))
    base = ix._pooled_block_keys(cache, keys, pos, 3, 1)
    assert type(cache.derived["qsa_pooled"]) is _QSAPooledFrontier
    monkeypatch.setenv("VMLX_QWEN4_QSA_POOL_CAPACITY", "1")
    grown = ix._pooled_block_keys(cache, keys, pos, 3, 1)
    fr = cache.derived["qsa_pooled"]
    assert type(fr) is _QSACapacityPooledFrontier
    assert cache.derived_nbytes == fr._buffer.nbytes > grown.nbytes
    cap = 3 * 16 * 4
    monkeypatch.setenv("VMLX_QWEN4_QSA_POOL_RETAIN_MAX_MB", str(cap / 1024**2))
    tight = ix._pooled_block_keys(cache, keys, pos, 3, 1)
    mx.eval(base, grown, tight)
    assert mx.array_equal(base, grown) and mx.array_equal(base, tight)
    assert fr.evicted == 1 and fr.nbytes == cap
    monkeypatch.setenv("VMLX_QWEN4_QSA_POOL_CAPACITY", "0")
    again = ix._pooled_block_keys(cache, keys, pos, 3, 1)
    assert type(cache.derived["qsa_pooled"]) is _QSAPooledFrontier
    assert mx.array_equal(base, again)
