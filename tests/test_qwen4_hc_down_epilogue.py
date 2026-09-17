"""Native dense HC projection and low-precision epilogue admission."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from vmlx_engine.metal import qwen4_hc_down_epilogue as candidate
from vmlx_engine.metal.qwen4_exact_down import _compatible_runtime
from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope


def words(actual, expected):
    mx.eval(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a, b = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_array_equal(a.view(np.uint16), b.view(np.uint16))


@pytest.mark.parametrize("combine", [False, True])
@pytest.mark.parametrize("scale", [.125, 1., 4.])
def test_native_dense_projection_and_epilogues(combine, scale):
    if not _compatible_runtime():
        pytest.skip("requires pinned M5 Max/MLX runtime")
    mx.random.seed(234)
    projection = nn.Linear(10240, 324 if combine else 320, bias=False)
    projection.weight = (mx.random.normal(projection.weight.shape)*.015).astype(mx.float16)
    projection.eval()
    n = (mx.random.normal((1, 1, 10240))*scale).astype(mx.float16)
    mx.eval(projection.parameters(), n)
    up = projection(n)
    expected = nn.silu(up[..., :320]/4)
    results = candidate._run(projection, n, combine=combine, capture=True)
    words(results[-1], up)
    words(results[0], expected)
    if combine:
        words(results[1], 2*mx.sigmoid(up[..., 320:]/4))
    with affine_moe_ar_scope():
        observed = candidate.hc_down_epilogue(projection, n, combine=combine, enabled=True)
    assert observed is not None
    words(observed[0], expected)


def test_default_off_and_unsupported_projection(monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_HC_DOWN_EPILOGUE", raising=False)
    assert not candidate.hc_down_epilogue_requested()
    p = nn.Linear(10240, 324, bias=False)
    p.set_dtype(mx.float16)
    p.eval()
    n = mx.zeros((1, 1, 10240), mx.float16)
    assert candidate.eligible(p, n, combine=True)
    assert not candidate.eligible(p, n, combine=False)
    assert not candidate.eligible(p, n.astype(mx.bfloat16), combine=True)
    assert not candidate.eligible(p, mx.concatenate([n, n], axis=1), combine=True)
    assert candidate.hc_down_epilogue(p, n, combine=True, enabled=True) is None
    q = nn.QuantizedLinear.from_linear(p, group_size=64, bits=8)
    assert not candidate.eligible(q, n, combine=True)
    p.train()
    assert not candidate.eligible(p, n, combine=True)


@pytest.mark.parametrize("combine", [False, True])
@pytest.mark.parametrize("from_normed", [False, True])
@pytest.mark.parametrize("up_enabled", [False, True])
def test_compiled_dispatch_and_up_composition(monkeypatch, combine, from_normed, up_enabled):
    if not _compatible_runtime():
        pytest.skip("requires pinned M5 Max/MLX runtime")
    from vmlx_engine.models.qwen4_exp.language import (
        GatedResidual, Qwen4ExpTextArgs, compile_hyper_connections,
        fuse_hyper_connection_projections,
    )
    monkeypatch.setenv("VMLX_QWEN4_HC_UP_MIX", "1" if up_enabled else "0")
    monkeypatch.setenv("VMLX_QWEN4_HC_DOWN_EPILOGUE", "1")
    module = GatedResidual(Qwen4ExpTextArgs(), use_combine=combine)
    module.set_dtype(mx.float16)
    module.eval()
    fuse_hyper_connection_projections(module)
    compile_hyper_connections(module)
    mx.random.seed(908)
    x = mx.random.normal((1, 1, 10240)).astype(mx.float16)
    n = module.hc_norm(x)
    mx.eval(module.parameters(), x, n)
    forward = (lambda: module.from_normed(x, n)) if from_normed else (lambda: module(x))
    stock = forward()
    mx.eval(stock)
    # An existing up-only AR trace must not swallow the new down candidate.
    if up_enabled:
        module._hc_down_epilogue = False
        with affine_moe_ar_scope():
            up_only = forward()
            mx.eval(up_only)
        module._hc_down_epilogue = True
    calls, run = [], candidate._run
    def observed(*args, **kwargs):
        calls.append(True)
        return run(*args, **kwargs)
    monkeypatch.setattr(candidate, "_run", observed)
    with affine_moe_ar_scope():
        fused = forward()
        mx.eval(fused)
    assert calls
    for a, b in zip(fused, stock) if combine else [(fused, stock)]:
        words(a, b)
    def forbidden(*args):
        pytest.fail("AR graph selected outside productive scope")
    setattr(module, module._candidate_graph_name(from_normed=from_normed), forbidden)
    for a, b in zip(forward(), stock) if combine else [(forward(), stock)]:
        words(a, b)
