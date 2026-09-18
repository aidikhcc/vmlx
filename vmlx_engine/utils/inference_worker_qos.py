# SPDX-License-Identifier: Apache-2.0
"""Opt-in experiment for the persistent multimodal inference worker.

No scheduling change is made by default.  The initializer runs on the existing
loader/step thread; it does not replace that executor or alter MLX streams.
Requested pthread QoS is not a measurement of effective priority or clock speed.
"""

import ctypes
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)
_ENV = "VMLX_MLLM_WORKER_QOS"
_USER_INITIATED = 0x19  # QOS_CLASS_USER_INITIATED, Darwin sys/qos.h


def configure_mllm_worker_qos() -> None:
    """Initialize only this worker, and fail visibly if an explicit trial fails."""
    mode = os.environ.get(_ENV, "").strip().lower()
    if mode in ("", "inherit"):
        return
    if mode not in ("observe", "user_initiated"):
        raise ValueError(f"{_ENV} must be inherit, observe, or user_initiated")
    if sys.platform != "darwin":
        raise RuntimeError(f"{_ENV}={mode} requires Darwin pthread QoS")

    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    current = libc.pthread_self
    current.argtypes = []
    current.restype = ctypes.c_void_p
    get_qos = libc.pthread_get_qos_class_np
    get_qos.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_int),
    ]
    get_qos.restype = ctypes.c_int
    worker = current()

    def read_qos() -> tuple[int, int]:
        qos, relative = ctypes.c_uint(), ctypes.c_int()
        result = get_qos(worker, ctypes.byref(qos), ctypes.byref(relative))
        if result != 0:
            raise RuntimeError(f"pthread_get_qos_class_np failed: errno={result}")
        return qos.value, relative.value

    before = read_qos()
    if mode == "user_initiated":
        set_qos = libc.pthread_set_qos_class_self_np
        set_qos.argtypes = [ctypes.c_uint, ctypes.c_int]
        set_qos.restype = ctypes.c_int
        result = set_qos(_USER_INITIATED, 0)
        if result != 0:
            raise RuntimeError(f"pthread_set_qos_class_self_np failed: errno={result}")
    after = read_qos()
    if mode == "user_initiated" and after != (_USER_INITIATED, 0):
        raise RuntimeError(f"Inference worker QoS read-back mismatch: {after!r}")
    if mode == "observe" and before != after:
        raise RuntimeError(f"Inference worker QoS changed during observation: {before!r} -> {after!r}")
    logger.info(
        "Inference worker QoS trial: mode=%s pid=%d thread=%s native_id=%d "
        "before_class=%d before_relative=%d after_class=%d after_relative=%d",
        mode, os.getpid(), threading.current_thread().name,
        threading.get_native_id(), *before, *after,
    )
