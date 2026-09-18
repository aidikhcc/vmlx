"""The optional QoS trial must not change default scheduling or worker ownership."""

import ctypes
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vmlx_engine.utils import inference_worker_qos as qos


@pytest.fixture
def fake_lib(monkeypatch):
    state = {"class": 0, "relative": 0}

    def get_qos(worker, qos_out, relative_out):
        assert worker == 0x123456789ABC  # must not truncate pthread_t to int32
        ctypes.cast(qos_out, ctypes.POINTER(ctypes.c_uint))[0] = state["class"]
        ctypes.cast(relative_out, ctypes.POINTER(ctypes.c_int))[0] = state["relative"]
        return 0

    def set_qos(value, relative):
        state.update({"class": value, "relative": relative})
        return 0

    lib = SimpleNamespace(
        pthread_self=Mock(return_value=0x123456789ABC),
        pthread_get_qos_class_np=Mock(side_effect=get_qos),
        pthread_set_qos_class_self_np=Mock(side_effect=set_qos),
    )
    monkeypatch.setattr(qos.sys, "platform", "darwin")
    monkeypatch.setattr(qos.ctypes, "CDLL", Mock(return_value=lib))
    return lib


@pytest.mark.parametrize("value", [None, "", "inherit", " INHERIT "])
def test_default_does_not_open_lib_or_change_scheduling(monkeypatch, value):
    monkeypatch.delenv(qos._ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(qos._ENV, value)
    cdll = Mock(side_effect=AssertionError("default may not query or change QoS"))
    monkeypatch.setattr(qos.ctypes, "CDLL", cdll)
    qos.configure_mllm_worker_qos()
    cdll.assert_not_called()


def test_observe_queries_this_thread_without_mutation(monkeypatch, fake_lib, caplog):
    monkeypatch.setenv(qos._ENV, "observe")
    with caplog.at_level(logging.INFO):
        qos.configure_mllm_worker_qos()
    fake_lib.pthread_set_qos_class_self_np.assert_not_called()
    assert fake_lib.pthread_get_qos_class_np.call_count == 2
    assert fake_lib.pthread_self.restype is ctypes.c_void_p
    assert fake_lib.pthread_get_qos_class_np.argtypes == [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)
    ]
    assert "mode=observe" in caplog.text and "after_class=0" in caplog.text


def test_user_initiated_reads_back_requested_class(monkeypatch, fake_lib, caplog):
    monkeypatch.setenv(qos._ENV, "user_initiated")
    with caplog.at_level(logging.INFO):
        qos.configure_mllm_worker_qos()
    fake_lib.pthread_set_qos_class_self_np.assert_called_once_with(0x19, 0)
    assert fake_lib.pthread_set_qos_class_self_np.argtypes == [ctypes.c_uint, ctypes.c_int]
    assert "before_class=0" in caplog.text and "after_class=25" in caplog.text


@pytest.mark.parametrize("mode", ["observe", "user_initiated"])
def test_unsupported_platform_fails_explicit_trial(monkeypatch, mode):
    monkeypatch.setenv(qos._ENV, mode)
    monkeypatch.setattr(qos.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="requires Darwin"):
        qos.configure_mllm_worker_qos()


def test_invalid_mode_fails_instead_of_silent_fallback(monkeypatch):
    monkeypatch.setenv(qos._ENV, "fastest")
    with pytest.raises(ValueError, match="must be inherit"):
        qos.configure_mllm_worker_qos()


def test_get_error_fails_before_setting(monkeypatch, fake_lib):
    monkeypatch.setenv(qos._ENV, "user_initiated")
    fake_lib.pthread_get_qos_class_np.side_effect = None
    fake_lib.pthread_get_qos_class_np.return_value = 22
    with pytest.raises(RuntimeError, match="get_qos_class_np failed: errno=22"):
        qos.configure_mllm_worker_qos()
    fake_lib.pthread_set_qos_class_self_np.assert_not_called()


def test_set_error_is_not_success(monkeypatch, fake_lib):
    monkeypatch.setenv(qos._ENV, "user_initiated")
    fake_lib.pthread_set_qos_class_self_np.side_effect = None
    fake_lib.pthread_set_qos_class_self_np.return_value = 1
    with pytest.raises(RuntimeError, match="set_qos_class_self_np failed: errno=1"):
        qos.configure_mllm_worker_qos()


def test_readback_mismatch_fails(monkeypatch, fake_lib):
    monkeypatch.setenv(qos._ENV, "user_initiated")
    fake_lib.pthread_set_qos_class_self_np.side_effect = None
    fake_lib.pthread_set_qos_class_self_np.return_value = 0
    with pytest.raises(RuntimeError, match="read-back mismatch"):
        qos.configure_mllm_worker_qos()


def test_initializer_preserves_single_worker_for_load_step_cleanup(monkeypatch, fake_lib):
    monkeypatch.setenv(qos._ENV, "observe")
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="mllm-worker",
        initializer=qos.configure_mllm_worker_qos,
    ) as executor:
        ids = [executor.submit(threading.get_native_id).result(timeout=2) for _ in range(3)]
    assert len(set(ids)) == 1
    assert ids[0] != threading.get_native_id()
    fake_lib.pthread_self.assert_called_once()
