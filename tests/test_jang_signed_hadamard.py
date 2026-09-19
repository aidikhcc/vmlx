"""Only the changed signed transform; no model download or full-suite run."""
import mlx.core as mx
import numpy as np
import pytest

from vmlx_engine.utils.jang_hadamard import hadamard_activation
from vmlx_engine.metal.jang_signed_hadamard import signed_hadamard

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


@pytest.mark.parametrize("block", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("rows,strided", [(1, False), (7, True), (128, False)])
@pytest.mark.parametrize("mode", ["1", "2"])
def test_words_match_reference(monkeypatch, block, dtype, inverse, rows, strided, mode):
    mx.random.seed(1909)
    x = mx.random.normal((rows, 3 * block * (2 if strided else 1))).astype(dtype)
    if strided:
        x = x[:, ::2]
    signs = mx.where(mx.random.uniform(shape=(3 * block,)) < 0.5, -1.0, 1.0)
    monkeypatch.setenv("VMLX_BONSAI_FUSED_RHT", "0")
    ref = hadamard_activation(x, block, signs, inverse=inverse)
    monkeypatch.setenv("VMLX_BONSAI_FUSED_RHT", mode)
    got = hadamard_activation(x, block, signs, inverse=inverse)
    direct = signed_hadamard(x, signs, block, inverse=inverse, prepared=mode == "2")
    assert direct is not None  # A fallback is not kernel qualification.
    mx.eval(ref, got)
    assert ref.dtype == got.dtype == dtype
    assert np.asarray(ref).tobytes() == np.asarray(got).tobytes()
    assert np.asarray(ref).tobytes() == np.asarray(direct).tobytes()


def test_ineligible_uses_reference(monkeypatch):
    x = mx.ones((1, 1024), dtype=mx.bfloat16)
    s = mx.ones((1024,), dtype=mx.float32)
    assert signed_hadamard(x, s, 1024) is None
    monkeypatch.setenv("VMLX_BONSAI_FUSED_RHT", "1")
    ref = hadamard_activation(x, 1024, s)
    assert ref.dtype == mx.bfloat16
    assert float(ref[0, 0]) == 32.0


def test_shape_validation_still_fail_closed(monkeypatch):
    monkeypatch.setenv("VMLX_BONSAI_FUSED_RHT", "1")
    with pytest.raises(ValueError, match="match the activation width"):
        hadamard_activation(mx.ones((1, 1024)), 1024, mx.ones((512,)))
    with pytest.raises(ValueError, match="does not divide"):
        hadamard_activation(mx.ones((1, 1025)), 1024, mx.ones((1025,)))


@pytest.mark.parametrize("inverse", [False, True])
def test_signed_zero_and_finite_extremes(monkeypatch, inverse):
    pattern = mx.array([0.0, -0.0, 65504.0, -65504.0, 2**-24, -(2**-24), 1.0, -1.0], dtype=mx.float16)
    x = mx.tile(pattern, 128).reshape(1, 1024)
    signs = mx.where(mx.arange(1024) % 3, -1.0, 1.0)
    monkeypatch.setenv("VMLX_BONSAI_FUSED_RHT", "0")
    ref = hadamard_activation(x, 1024, signs, inverse=inverse)
    got = signed_hadamard(x, signs, 1024, inverse=inverse)
    assert got is not None
    assert np.asarray(ref).tobytes() == np.asarray(got).tobytes()


def test_prefill_only_mode_does_not_enter_kernel_for_decode(monkeypatch):
    import vmlx_engine.metal.jang_signed_hadamard as module
    called = []
    actual = module.signed_hadamard
    def observe(x, signs, block, **kwargs):
        called.append(tuple(x.shape))
        return actual(x, signs, block, **kwargs)
    monkeypatch.setattr(module, "signed_hadamard", observe)
    monkeypatch.setenv("VMLX_BONSAI_FUSED_RHT", "3")
    signs = mx.where(mx.arange(5120) % 3, 1.0, -1.0)
    mx.eval(hadamard_activation(mx.ones((1, 1, 5120), dtype=mx.float16), 1024, signs))
    assert not called
    value = mx.ones((1, 128, 5120), dtype=mx.float16)
    got = hadamard_activation(value, 1024, signs)
    assert called == [(1, 128, 5120)]
    monkeypatch.setenv("VMLX_BONSAI_FUSED_RHT", "0")
    ref = hadamard_activation(value, 1024, signs)
    assert np.asarray(got).tobytes() == np.asarray(ref).tobytes()
