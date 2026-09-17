"""Native arithmetic and admission for the default-off Qwen HC up/mix candidate."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from vmlx_engine.metal import qwen4_hc_up_mix as candidate
from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope


def fixture(bits=8, group=64, seed=11):
    mx.random.seed(seed)
    linear = nn.Linear(320, 10240, bias=False)
    linear.weight = (mx.random.normal((10240, 320)) * 0.04).astype(mx.float16)
    projection = (linear if bits is None else
                  nn.QuantizedLinear.from_linear(linear, group_size=group, bits=bits))
    projection.eval()
    activated = mx.random.normal((1, 1, 320)).astype(mx.float16)
    normed = mx.random.normal((1, 1, 10240)).astype(mx.float16)
    mx.eval(projection.parameters(), activated, normed)
    return projection, activated, normed


def words_equal(actual, expected):
    mx.eval(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a, b = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_array_equal(a.view(np.uint16), b.view(np.uint16))


@pytest.mark.parametrize("bits,group", [(None, 64), (8, 64), (8, 32), (6, 64), (4, 64), (3, 64), (2, 64)])
def test_native_projection_sigmoid_and_mean(bits, group):
    if not candidate._compatible_runtime():
        pytest.skip("Numerical qualification requires the pinned M5 Max/MLX runtime")
    p, a, n = fixture(bits, group)
    assert candidate.eligible(p, a, n, hc_count=4, hidden_size=2560)
    up = p(a)
    gate = mx.sigmoid(up)
    expected = (gate.reshape(1, 1, 4, 2560) * n.reshape(1, 1, 4, 2560)).mean(-2)
    mixed, captured_up, captured_gate = candidate._run(p, a, n, capture=True)
    words_equal(captured_up, up)
    words_equal(captured_gate, gate)
    words_equal(mixed, expected)
    with affine_moe_ar_scope():
        actual = candidate.hc_up_mix(p, a, n, hc_count=4, hidden_size=2560, enabled=True)
    assert actual is not None
    words_equal(actual, expected)


def test_default_off_and_unsupported_shapes(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_HC_UP_MIX", raising=False)
    assert not candidate.hc_up_mix_requested()
    p, a, n = fixture()
    assert candidate.hc_up_mix(p, a, n, hc_count=4, hidden_size=2560, enabled=False) is None
    assert candidate.hc_up_mix(p, a, n, hc_count=4, hidden_size=2560, enabled=True) is None
    assert not candidate.eligible(p, mx.concatenate([a, a], axis=1), n, hc_count=4, hidden_size=2560)
    assert not candidate.eligible(p, a.astype(mx.bfloat16), n, hc_count=4, hidden_size=2560)
    assert not candidate.eligible(p, a, n, hc_count=2, hidden_size=2560)
    p.train()
    assert not candidate.eligible(p, a, n, hc_count=4, hidden_size=2560)


@pytest.mark.parametrize("from_normed", [False, True])
@pytest.mark.parametrize("combine", [False, True])
@pytest.mark.parametrize("bits", [None, 8])
def test_compiled_stock_and_ar_graphs_are_separate(monkeypatch, from_normed, combine, bits):
    if not candidate._compatible_runtime():
        pytest.skip("requires pinned M5 Max/MLX runtime")
    from vmlx_engine.models.qwen4_exp.language import (
        GatedResidual, Qwen4ExpTextArgs, compile_hyper_connections,
        fuse_hyper_connection_projections,
    )
    monkeypatch.setenv("VMLX_QWEN4_HC_UP_MIX", "1")
    module = GatedResidual(Qwen4ExpTextArgs(), use_combine=combine)
    module.set_dtype(mx.float16)
    if bits is not None:
        nn.quantize(module, group_size=64, bits=bits)
    module.eval()
    fuse_hyper_connection_projections(module)
    compile_hyper_connections(module)
    mx.random.seed(907)
    x = mx.random.normal((1, 1, 10240)).astype(mx.float16)
    normed = module.hc_norm(x)
    mx.eval(module.parameters(), x, normed)
    forward = (lambda: module.from_normed(x, normed)) if from_normed else (lambda: module(x))
    # Trace stock first with the same shape, as a single-token seed may do.
    expected = forward()
    mx.eval(expected)
    calls = []
    run = candidate._run
    def tracked(*args, **kwargs):
        calls.append(True)
        return run(*args, **kwargs)
    monkeypatch.setattr(candidate, "_run", tracked)
    with affine_moe_ar_scope():
        actual = forward()
        mx.eval(actual)
    assert calls  # actual candidate graph, not a cached stock trace
    for got, want in zip(actual, expected) if combine else [(actual, expected)]:
        words_equal(got, want)
    # A subsequent non-AR call must not select the cached candidate graph.
    selected = ("_compiled_normed_up_mix_forward" if from_normed
                else "_compiled_up_mix_forward")
    def forbidden(*args):
        pytest.fail("candidate graph selected outside productive AR")
    setattr(module, selected, forbidden)
    again = forward()
    for got, want in zip(again, expected) if combine else [(again, expected)]:
        words_equal(got, want)
    with affine_moe_ar_scope():
        assert not module._use_up_mix(mx.concatenate([x, x], axis=0))
        assert not module._use_up_mix(mx.concatenate([x, x], axis=1))
        module.train()
        assert not module._use_up_mix(x)
