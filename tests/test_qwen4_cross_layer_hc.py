"""Cross-layer handoff and excluded-path controls; no throughput claims."""

from contextlib import nullcontext

import mlx.core as mx
import pytest

from tests.test_qwen4_eager_dispatch import (
    _assert_exact, _cache_snapshot, _model, _snapshot,
)
from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope
from vmlx_engine.models.qwen4_exp import language


def _reference_combine_norm(residual, block, inject, weight, *, eps, group_size, enabled):
    # Tiny-model control-flow oracle, NOT a replacement for actual-shape Metal
    # kernel parity or the actual-bundle full-logit/native-state qualification.
    assert enabled
    shape = residual.shape
    streams = shape[-1] // group_size
    product = (block[..., None, :] * inject[..., :, None]).reshape(shape)
    combined = (residual + product).astype(residual.dtype)
    normed = mx.fast.rms_norm(combined.reshape(*shape[:-1], streams, group_size), None, eps)
    return combined, normed.reshape(shape) * weight


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True)])
def test_default_off_captured_at_construction(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VMLX_QWEN4_HC_CROSS_LAYER", raising=False)
    else:
        monkeypatch.setenv("VMLX_QWEN4_HC_CROSS_LAYER", value)
    model = _model(monkeypatch)
    assert model.model._cross_layer_hc is expected
    monkeypatch.setenv("VMLX_QWEN4_HC_CROSS_LAYER", "0" if expected else "1")
    assert model.model._cross_layer_hc is expected


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("native_kernel", [False, True])
def test_connected_exact_state_expanded_hidden_and_caller_stream(monkeypatch, eager, native_kernel):
    model = _model(monkeypatch, enabled=eager)
    if not native_kernel:
        monkeypatch.setattr(language, "hc_combine_norm", _reference_combine_norm)
    submissions = []
    submit = mx.async_eval

    def record(*values):
        submissions.append((len(values), mx.default_stream(mx.gpu)))
        submit(*values)

    monkeypatch.setattr(mx, "async_eval", record)
    stream = mx.new_stream(mx.gpu)
    arms = []
    with mx.stream(stream), affine_moe_ar_scope():
        for enabled in (False, True):
            model.model._cross_layer_hc = enabled
            cache = model.make_cache()
            arm = []
            for ids in ([11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67],
                        [71], [73, 79, 83], [89]):
                logits, hidden = model(mx.array([ids]), cache=cache, return_hidden=True)
                arm.append((_snapshot(logits), _snapshot(hidden), _cache_snapshot(cache)))
            arms.append(tuple(arm))
    _assert_exact(*arms)
    # 8 layers, no handoff before layer1's PLE, no handoff after the last.
    expected = 0 if native_kernel else 12
    assert model.model._cross_layer_hc_graph_calls == expected
    assert sum(n == 2 for n, _ in submissions) == (expected if eager else 0)
    assert all(s == stream for _, s in submissions)
    # No carried tensor belongs to the module or serialized cache.
    assert "attn_normed" not in model.model


@pytest.mark.parametrize("excluded", [
    "disabled", "outside_ar", "batch", "prefill", "verify", "checkpoint",
    "profile", "fingerprint", "training", "consumer_opt_out",
])
def test_excluded_paths_do_not_handoff(monkeypatch, excluded):
    model = _model(monkeypatch, enabled=False)
    model.model._cross_layer_hc = excluded != "disabled"
    monkeypatch.setattr(language, "hc_combine_norm", _reference_combine_norm)
    ids, kwargs = [[11]], {}
    if excluded == "batch":
        ids = [[11], [17]]
    elif excluded in ("prefill", "checkpoint"):
        ids = [[11, 17, 23, 29, 31]]
        if excluded == "checkpoint":
            kwargs["prefill_checkpoint_steps"] = (2, 4)
    elif excluded == "verify":
        kwargs["n_confirmed"] = 1
    elif excluded == "profile":
        monkeypatch.setattr(language, "_layer_profile_enabled", lambda _: True)
    elif excluded == "fingerprint":
        monkeypatch.setattr(language, "_layer_fingerprint_enabled", lambda _: True)
        monkeypatch.setattr(language, "_log_layer_fingerprint", lambda *args: None)
        monkeypatch.setattr(language, "_log_module_state", lambda *args: None)
    elif excluded == "training":
        model.train()
    elif excluded == "consumer_opt_out":
        for layer in model.layers:
            layer.attn_hyper_connection._combine_norm = False
    scope = nullcontext() if excluded == "outside_ar" else affine_moe_ar_scope()
    with scope:
        mx.eval(model(mx.array(ids), cache=model.make_cache(), **kwargs).logits)
    assert model.model._cross_layer_hc_graph_calls == 0


def test_no_pre_normalized_input_across_ple_or_speculative_boundary(monkeypatch):
    model = _model(monkeypatch, enabled=False)
    hidden = mx.ones((1, 1, model.args.hidden_size * model.args.hc_count))
    with affine_moe_ar_scope():
        with pytest.raises(ValueError, match="Pre-normalized HC input"):
            model.layers[1](hidden, _attn_normed=hidden)
        with pytest.raises(ValueError, match="Pre-normalized HC input"):
            model.layers[0](hidden, _attn_normed=hidden, n_confirmed=1)


def test_admitted_failure_is_not_replayed_or_carried_to_next_forward(monkeypatch):
    model = _model(monkeypatch, enabled=False)
    model.model._cross_layer_hc = True
    target = model.layers[2].attn_hyper_connection.hc_norm.weight
    failure = RuntimeError("cross-layer kernel failed")
    calls = []

    def fail_once(residual, block, inject, weight, **kwargs):
        if weight is target:
            calls.append(True)
            raise failure
        return _reference_combine_norm(residual, block, inject, weight, **kwargs)

    monkeypatch.setattr(language, "hc_combine_norm", fail_once)
    with affine_moe_ar_scope(), pytest.raises(RuntimeError) as caught:
        model(mx.array([[11]]), cache=model.make_cache())
    assert caught.value is failure and calls == [True]
    monkeypatch.setattr(language, "hc_combine_norm", _reference_combine_norm)
    with affine_moe_ar_scope():
        mx.eval(model(mx.array([[17]]), cache=model.make_cache()).logits)
    assert model.model._cross_layer_hc_graph_calls == 6
