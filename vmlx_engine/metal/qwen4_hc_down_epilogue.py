# Copyright © 2023-2024 Apple Inc.
# Native GEMV traversal adapted from MLX v0.32.2 gemv.h, MIT License.
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
"""Opt-in native dense HC down/injection projection with rounded epilogues.

Keep the MLX 0.32.2 K10240/N320-or-324 BN8 GEMV traversal, projection rounding,
half division, SiLU, and injection sigmoid. This is not the upstream quantized
BF16 split-K algorithm. Unsupported formats use the existing projection graph.
"""
from functools import lru_cache
import logging
import os

import mlx.core as mx
import mlx.nn as nn

from .qwen4_hc_up_mix import hc_up_mix_scope_eligible

logger = logging.getLogger(__name__)


def hc_down_epilogue_requested():
    return os.environ.get("VMLX_QWEN4_HC_DOWN_EPILOGUE", "0") == "1"


def eligible(projection, normed, *, combine):
    return (type(projection) is nn.Linear and not projection.training
            and "bias" not in projection
            and normed.shape == (1, 1, 10240) and normed.dtype == mx.float16
            and projection.weight.dtype == mx.float16
            and projection.weight.shape == (324 if combine else 320, 10240))


_SOURCE = r'''
    constexpr uint K = 10240;
    constexpr uint ROWS = 4;
    uint first = threadgroup_position_in_grid.x * ROWS;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    float accum[ROWS] = {0};
    for (uint base = 0; base < K; base += 1024) {
        uint k = base + sg * 128 + lane * 4;
        float values[4];
        for (uint n = 0; n < 4; ++n) values[n] = float(normed[k+n]);
        for (uint r = 0; r < ROWS; ++r) {
            for (uint n = 0; n < 4; ++n)
                accum[r] += weight[(first+r)*K+k+n] * values[n];
        }
    }
    threadgroup float partial[8 * ROWS];
    for (uint r = 0; r < ROWS; ++r) {
        for (ushort sn = 16; sn >= 1; sn >>= 1)
            accum[r] += simd_shuffle_down(accum[r], sn);
        if (lane == 0) partial[sg*ROWS+r] = accum[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        for (uint r = 0; r < ROWS; ++r) {
            for (uint s = 1; s < 8; ++s) accum[r] += partial[s*ROWS+r];
            T projected = T(accum[r]);
            T scaled = T(projected / T(4));
            auto y = 1 / (1 + metal::exp(metal::abs(scaled)));
            T gate = (scaled < 0) ? y : 1 - y;
            if (first+r < 320) activated[first+r] = T(scaled * gate);
            /* INJECTION */
            /* CAPTURE */
        }
    }
'''


@lru_cache(maxsize=4)
def _kernel(combine, capture):
    source = _SOURCE.replace("/* INJECTION */", (
        "else injection[first+r-320] = T(T(2) * gate);" if combine else ""))
    source = source.replace("/* CAPTURE */", (
        "projected_values[first+r] = projected;" if capture else ""))
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_native_hc_down_epilogue_v1"
             + ("_combined" if combine else "_final")
             + ("_capture" if capture else ""),
        input_names=["normed", "weight"],
        output_names=["activated"] + (["injection"] if combine else [])
                     + (["projected_values"] if capture else []),
        source=source, ensure_row_contiguous=True)


def _run(projection, normed, *, combine, capture=False):
    rows = 324 if combine else 320
    return _kernel(combine, capture)(
        inputs=[normed, projection.weight], template=[("T", normed.dtype)],
        grid=(rows//4*256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(1, 1, 320)] + ([(1, 1, 4)] if combine else [])
                      + ([(1, 1, rows)] if capture else []),
        output_dtypes=[normed.dtype] * (1+int(combine)+int(capture)))


_OBSERVED = False


def hc_down_epilogue(projection, normed, *, combine, enabled):
    """Decline metadata before dispatch; never retry a failed productive graph."""
    global _OBSERVED
    if (not hc_up_mix_scope_eligible(normed, hc_count=4, hidden_size=2560,
                                    enabled=enabled)
            or not eligible(projection, normed, combine=combine)):
        return None
    outputs = _run(projection, normed, combine=combine)
    if not _OBSERVED:
        logger.info("Qwen HC down/epilogue graph: dense FP16 K10240 BN8 "
                    "combine=%s scope=productive_ar candidate=true", combine)
        _OBSERVED = True
    return outputs[0], outputs[1] if combine else None
