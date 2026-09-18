"""Next-chunk PLE reads change scheduling, not native state or chunk policy."""

import threading

import mlx.core as mx
import numpy as np
import pytest

from tests.test_qwen4_eager_dispatch import _assert_exact, _cache_snapshot, _snapshot
from tests.test_qwen4_ple_host_gather import _bits, _table
from tests.test_qwen4_ple_prefetch import _file_model
from vmlx_engine.models.qwen4_exp import table_reader


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_chunk_ticket_has_separate_policy_and_bounded_exact_packed_data(tmp_path, monkeypatch, dtype):
    monkeypatch.delenv("VMLX_QWEN4_PLE_PREFETCH", raising=False)
    monkeypatch.delenv("VMLX_QWEN4_PLE_CHUNK_LOOKAHEAD", raising=False)
    table = _table(tmp_path, [(1, 32), (2, 64), (6, 32), (8, 128)], dtype)
    rows = np.resize(np.array([0, 7, 21, 14, 25]), 8193)
    try:
        assert table.prefetch_rows(rows, lookahead=True) is None
        table._chunk_lookahead_enabled = True
        expected = _bits(table.gather_mlx(rows)).copy()
        assert table.prefetch_rows(rows) is None  # AR switch remains OFF.
        ticket = table.prefetch_rows(rows, lookahead=True)
        assert ticket is not None
        actual = table.gather_mlx(rows, prepared=ticket)
        np.testing.assert_array_equal(_bits(actual), expected)
        assert table.prefetch_rows(np.zeros(65537), lookahead=True) is None
        monkeypatch.setattr(table_reader, "_PREFETCH_MAX_PACKED_BYTES", 1)
        assert table.prefetch_rows(rows, lookahead=True) is None
        assert table._prefetch_ticket is None
    finally:
        table.close()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("restored", [False, True])
def test_connected_chunks_are_exact_with_native_history(tmp_path, monkeypatch, dtype, restored):
    monkeypatch.setenv("VMLX_QWEN4_PLE_CHUNK_LOOKAHEAD", "1")
    model, _, table = _file_model(tmp_path, monkeypatch, dtype)
    model.model._ple_prefetch = False
    tokens = mx.array([[11 + (i % 47) for i in range(170)]])
    cuts = (0, 65, 138, 169, 170)
    results = []
    try:
        for enabled in (False, True):
            monkeypatch.setenv("VMLX_QWEN4_PLE_CHUNK_LOOKAHEAD", str(int(enabled)))
            cache = model.make_cache()
            if restored:
                mx.eval(model(mx.array([[17, 29, 47]]), cache=cache).logits)
            states = []
            with model.prefill_read_ahead() as ahead:
                for i, (start, end) in enumerate(zip(cuts, cuts[1:])):
                    ids = tokens[:, start:end]
                    borrowed = ahead.take(ids, cache) if ahead else None
                    output = model(ids, cache=cache, ple_prepared_reads=borrowed)
                    if ahead and i + 2 < len(cuts):
                        before = _snapshot(cache[1][2])
                        ahead.prepare(tokens[:, end:cuts[i + 2]], cache)
                        _assert_exact(before, _snapshot(cache[1][2]))
                    states.append((_snapshot(output.logits), _cache_snapshot(cache)))
            results.append(tuple(states))
            if ahead:
                assert ahead.closed and not ahead.pending
                assert ahead.stats == {"prepared": 2, "matched": 2, "discarded": 0}
        _assert_exact(*results)
        assert table._prefetch_ticket is None
    finally:
        table.close()


@pytest.mark.parametrize("mismatch", ["chunk", "history"])
def test_changed_chunk_or_history_drains_and_falls_back(tmp_path, monkeypatch, mismatch):
    monkeypatch.setenv("VMLX_QWEN4_PLE_CHUNK_LOOKAHEAD", "1")
    model, _, table = _file_model(tmp_path, monkeypatch, mx.float16)
    ids = mx.array([[11, 17, 23, 29]])
    cache = model.make_cache()
    try:
        mx.eval(model(mx.array([[19, 31]]), cache=cache).logits)
        with model.prefill_read_ahead() as ahead:
            ahead.prepare(ids, cache)
            if mismatch == "chunk":
                ids = ids[:, :2]
            else:
                cache[1][2] = mx.array([[41, 43]])
            before = _cache_snapshot(cache)
            assert ahead.take(ids, cache) == {}
            _assert_exact(before, _cache_snapshot(cache))
            assert table._prefetch_ticket is None
            assert ahead.stats["discarded"] == 1
    finally:
        table.close()


def test_request_error_drains_pending_read_before_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_PLE_CHUNK_LOOKAHEAD", "1")
    model, _, table = _file_model(tmp_path, monkeypatch, mx.float16)
    entered, release, finished = (threading.Event() for _ in range(3))
    read = table._read_host_assembled

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        try:
            return read(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(table, "_read_host_assembled", blocked)
    try:
        with pytest.raises(RuntimeError, match="request cancelled"):
            with model.prefill_read_ahead() as ahead:
                ahead.prepare(mx.array([[11, 17, 23]]), model.make_cache())
                assert entered.wait(5)
                release.set()
                raise RuntimeError("request cancelled")
        assert ahead.closed and finished.is_set()
        assert table._prefetch_ticket is None
    finally:
        release.set()
        table.close()


@pytest.mark.parametrize("fail", [False, True])
def test_production_chunk_loop_announces_before_eval_and_always_closes(monkeypatch, fail):
    from tests.test_hybrid_chunked_prefill_equivalence import _make_gate_fixture
    from vmlx_engine import mllm_batch_generator as gen

    monkeypatch.setenv("VMLX_ALLOW_HYBRID_CHUNKED_PREFILL", "1")
    generator, model, request = _make_gate_fixture(with_proven_config=True)
    generator.prefill_step_size = 2
    events = []

    class Scope:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_):
            events.append("close")

        def take(self, ids, cache):
            events.append(("take", ids.tolist()))
            return {}

        def prepare(self, ids, cache):
            events.append(("prepare", ids.tolist()))
            if fail:
                raise OSError("lookahead read failure")

    model.language_model.prefill_read_ahead = Scope
    materialize = gen._materialize_prefill_cache_state

    def evaluated(cache):
        events.append("eval")
        materialize(cache)

    monkeypatch.setattr(gen, "_materialize_prefill_cache_state", evaluated)
    try:
        if fail:
            with pytest.raises(OSError, match="lookahead read failure"):
                generator._run_vision_encoding(request, model.language_model.make_cache())
        else:
            result = generator._run_vision_encoding(request, model.language_model.make_cache())
            assert result.shape == (1, 1, 8)
            assert events[1:4] == [("take", [[1, 2]]), ("prepare", [[3, 4]]), "eval"]
            assert model.language_model.calls == [
                {"tokens": 2, "return_logits": False},
                {"tokens": 2, "return_logits": False},
                {"tokens": 1, "return_logits": False},
                {"tokens": 1, "return_logits": True},
            ]
        assert events[0] == "enter" and events[-1] == "close"
    finally:
        generator.close()
