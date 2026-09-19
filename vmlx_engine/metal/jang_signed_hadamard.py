"""Opt-in signed FP32 Hadamard fusion for the declared JANG contract.

The radix-16 schedule matches MLX 0.32.2's Metal hadamard_n. Input conversion,
the forward/inverse signs, normalization and output conversion are fused;
butterfly arithmetic stays FP32. No weights or cache state are transformed.

Schedule adapted from ml-explore/mlx, mlx/backend/metal/kernels/hadamard.h.
Copyright (c) 2024 Apple Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from functools import lru_cache
import math

import mlx.core as mx


@lru_cache(maxsize=8)
def _kernel(block: int, inverse: bool):
    log_n = block.bit_length() - 1
    # The scale is rounded to FP32 in the same place as stock Hadamard.
    scale = repr(1.0 / math.sqrt(block))
    source = r"""
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        constexpr short threads = N / 16;
        constexpr short stages = LOG_N / 4;
        constexpr short final_bits = LOG_N % 4;
        constexpr short final_radix = 1 << final_bits;
        const short lane = thread_position_in_threadgroup.x;
        const uint base = threadgroup_position_in_grid.y * N;
        threadgroup float shared[N];
        for (short j = 0; j < 4; ++j) {
            const short at = j * 4 * threads + lane * 4;
            for (short r = 0; r < 4; ++r) {
                float value = float(x[base + at + r]);
                if (!INVERSE) value = value * signs[(base + at + r) % W];
                shared[at + r] = value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float values[16];
        short stride = 1;
        for (short stage = 0; stage < stages; ++stage) {
            const short k = lane & (stride - 1);
            const short start = ((lane - k) << 4) + k;
            for (short r = 0; r < 16; ++r)
                values[r] = shared[start + stride * r];
            for (short span = 1; span < 16; span <<= 1) {
                for (short pair = 0; pair < 8; ++pair) {
                    const short low = pair & (span - 1);
                    const short a = ((pair - low) << 1) + low;
                    const float v0 = values[a], v1 = values[a + span];
                    values[a] = v0 + v1;
                    values[a + span] = v0 - v1;
                }
            }
            for (short r = 0; r < 16; ++r)
                shared[start + stride * r] = values[r];
            stride <<= 4;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (final_radix > 1) {
            for (short t = 0; t < 16 / final_radix; ++t) {
                const short index = lane + t * threads;
                const short k = index & (stride - 1);
                const short start = ((index - k) << final_bits) + k;
                for (short r = 0; r < final_radix; ++r)
                    values[r] = shared[start + stride * r];
                for (short span = 1; span < final_radix; span <<= 1) {
                    for (short pair = 0; pair < final_radix / 2; ++pair) {
                        const short low = pair & (span - 1);
                        const short a = ((pair - low) << 1) + low;
                        const float v0 = values[a], v1 = values[a + span];
                        values[a] = v0 + v1;
                        values[a + span] = v0 - v1;
                    }
                }
                for (short r = 0; r < final_radix; ++r)
                    shared[start + stride * r] = values[r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (short j = 0; j < 4; ++j) {
            const short at = j * 4 * threads + lane * 4;
            for (short r = 0; r < 4; ++r) {
                float value = shared[at + r] * SCALE;
                if (INVERSE) value = value * signs[(base + at + r) % W];
                out[base + at + r] = T(value);
            }
        }
    """
    source = source.replace("LOG_N", str(log_n)).replace("SCALE", scale + "f")
    return mx.fast.metal_kernel(
        name=f"jang_signed_hadamard_{block}_{int(inverse)}",
        input_names=["x", "signs"], output_names=["out"],
        source=source, ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def signed_hadamard(x: mx.array, signs: mx.array, block: int, *, inverse=False):
    """Return a candidate result or None for a shape/dtype outside this gate."""
    if (not mx.metal.is_available() or mx.default_device() != mx.gpu
            or block not in (512, 1024, 2048, 4096)
            or x.dtype not in (mx.float16, mx.float32)
            or signs.dtype != mx.float32 or not x.ndim or not x.size
            or x.shape[-1] % block or signs.shape != (x.shape[-1],)):
        return None
    return _kernel(block, inverse)(
        inputs=[x, signs],
        template=[("T", x.dtype), ("N", block), ("W", x.shape[-1]),
                  ("INVERSE", inverse)],
        grid=(block // 16, x.size // block, 1),
        threadgroup=(block // 16, 1, 1),
        output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]
