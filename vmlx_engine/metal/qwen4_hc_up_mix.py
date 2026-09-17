# Copyright © 2023-2024 Apple Inc.
# Dense GEMV arithmetic adapted from MLX v0.32.2 gemv.h under the MIT
# License. QMV helpers imported below retain the same notice.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Candidate native-order HC up-projection and stream mixing.

The missing boundary is demonstrated by oMLX PR #3469's hc_fused.py, but its
BF16/FP32 epilogue is not transplanted. This implementation keeps MLX 0.32.2's
K320 GEMV/general-QMV traversal and projection/sigmoid/product/sum rounding.
The native projections feeding this helper, grouped norm, residual injection,
cache state, and sampler are unchanged. Default OFF pending runtime evidence.

QMV load/dot helpers are the MIT-licensed Apple adaptation in qwen4_exact_down;
that module retains the complete notice. No upstream Apache code is copied.
"""

from functools import lru_cache
import logging
import os

import mlx.core as mx
import mlx.nn as nn

from .affine_moe_pair_decode import affine_moe_ar_scope_active
from .qwen4_exact_down import _HEADER, _compatible_runtime

logger = logging.getLogger(__name__)


def hc_up_mix_requested():
    return os.environ.get("VMLX_QWEN4_HC_UP_MIX", "0") == "1"


def hc_up_mix_scope_eligible(value, *, hc_count, hidden_size, enabled):
    """Check on every call, outside mx.compile's cached Python tracing.

    Projection/activation admission remains inside the dedicated candidate
    graph. An AR trace must never select this graph for prefill or MTP.
    """
    return (enabled and hc_count == 4 and hidden_size == 2560
            and value.shape == (1, 1, 10240) and value.dtype == mx.float16
            and affine_moe_ar_scope_active()
            and mx.default_device() == mx.gpu and mx.metal.is_available()
            and _compatible_runtime())


def eligible(projection, activated, normed, *, hc_count, hidden_size):
    """Metadata-only admission based on tensors, never the quant's model label."""
    if (hc_count != 4 or hidden_size != 2560
            or activated.shape != (1, 1, 320)
            or normed.shape != (1, 1, 10240)
            or activated.dtype != mx.float16 or normed.dtype != mx.float16
            or projection.training
            or "bias" in projection):
        return False
    if type(projection) is nn.Linear:
        return (projection.weight.shape == (10240, 320)
                and projection.weight.dtype == mx.float16)
    if (type(projection) is not nn.QuantizedLinear or projection.mode != "affine"
            or projection.bits not in (2, 3, 4, 6, 8)
            or projection.group_size not in (32, 64)):
        return False
    bits, group = projection.bits, projection.group_size
    return (projection.weight.dtype == mx.uint32
            and projection.weight.shape == (10240, 320 * bits // 32)
            and projection.scales.shape == (10240, 320 // group)
            and projection.biases.shape == projection.scales.shape
            and projection.scales.dtype == mx.float16
            and projection.biases.dtype == mx.float16)


_QMV = r'''
    constexpr uint V = BITS==2?16:(BITS==3||BITS==4?8:4);
    const device uint8_t* packed = (const device uint8_t*)weight;
    float accum[ROWS] = {0}, values[V];
    for (uint base = 0; base < K; base += 32 * V) {
        uint k = base + lane * V;
        if (k < K) {
            float sum = qwen_down_load<T, BITS, V>(activated + k, values);
            for (uint r = 0; r < ROWS; ++r) {
                size_t row = first + r;
                size_t offset = row * (K * BITS / 8) + k * BITS / 8;
                size_t meta = row * (K / GS) + k / GS;
                accum[r] += qwen_down_dot<BITS, V>(packed + offset, values,
                    float(scales[meta]), float(biases[meta]), sum);
            }
        }
    }
'''

_GEMV = r'''
    float accum[ROWS] = {0};
    for (uint base = 0; base < K; base += 128) {
        uint k = base + lane * 4;
        float values[4];
        for (uint n = 0; n < 4; ++n)
            values[n] = (k+n < K) ? float(activated[k+n]) : 0.0f;
        for (uint r = 0; r < ROWS; ++r) {
            for (uint n = 0; n < 4; ++n) {
                T w = (k+n < K) ? weight[(first+r)*K+k+n] : T(0);
                accum[r] += w * values[n];
            }
        }
    }
'''

_SOURCE = r'''
    constexpr uint H = 2560;
    constexpr uint K = 320;
    constexpr uint ROWS = 4;
    uint tile = threadgroup_position_in_grid.x;
    uint stream = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint first = stream * H + tile * ROWS;
    /* DOT */
    threadgroup T products[4 * ROWS];
    for (uint r = 0; r < ROWS; ++r) {
        /* REDUCE */
        if (lane == 0) {
            T projected = T(reduced);
            auto y = 1 / (1 + metal::exp(metal::abs(projected)));
            T gate = (projected < 0) ? y : 1 - y;
            products[stream * ROWS + r] = T(gate * normed[first + r]);
            /* CAPTURE */
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (stream == 0 && lane < ROWS) {
        T total = T(products[lane] + T(0));
        for (uint s = 1; s < 4; ++s)
            total = T(products[s * ROWS + lane] + total);
        mixed[tile * ROWS + lane] = T(total * T(0.25));
    }
'''


@lru_cache(maxsize=4)
def _kernel(capture=False, dense=False):
    source = _SOURCE.replace("/* DOT */", _GEMV if dense else _QMV)
    source = source.replace("/* REDUCE */", (
        "float reduced=accum[r]; for(ushort sn=16;sn>=1;sn>>=1) "
        "reduced+=simd_shuffle_down(reduced,sn);"
        if dense else "float reduced=simd_sum(accum[r]);"))
    source = source.replace("/* CAPTURE */", (
        "projected_values[first+r]=projected; gates[first+r]=gate;"
        if capture else ""))
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_native_hc_up_mix_v1" + ("_dense" if dense else "_affine")
             + ("_capture" if capture else ""),
        input_names=["activated", "normed", "weight"] + ([] if dense else ["scales", "biases"]),
        output_names=["mixed"] + (["projected_values", "gates"] if capture else []),
        header=_HEADER, source=source, ensure_row_contiguous=True)


def _run(projection, activated, normed, *, capture=False):
    dense = type(projection) is nn.Linear
    return _kernel(capture, dense)(
        inputs=[activated, normed, projection.weight]
               + ([] if dense else [projection.scales, projection.biases]),
        template=[("T", activated.dtype)] + ([] if dense else
                  [("BITS", projection.bits), ("GS", projection.group_size)]),
        grid=(640 * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(1, 1, 2560)] + ([(1, 1, 10240)] * 2 if capture else []),
        output_dtypes=[activated.dtype] * (3 if capture else 1))


_OBSERVED = False


def hc_up_mix(projection, activated, normed, *, hc_count, hidden_size, enabled):
    """Decline before execution; never replay a failed stateful forward."""
    global _OBSERVED
    if (not hc_up_mix_scope_eligible(normed, hc_count=hc_count,
                                    hidden_size=hidden_size, enabled=enabled)
            or not eligible(projection, activated, normed,
                            hc_count=hc_count, hidden_size=hidden_size)):
        return None
    output = _run(projection, activated, normed)[0]
    if not _OBSERVED:
        logger.info("Qwen HC up/mix graph: bits=%s group=%s dtype=%s K320 "
                    "streams4 hidden2560 scope=productive_ar candidate=true",
                    getattr(projection, "bits", "dense"),
                    getattr(projection, "group_size", None), activated.dtype)
        _OBSERVED = True
    return output
