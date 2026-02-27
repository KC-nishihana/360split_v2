import cv2

from core.accelerator import Accelerator


def _fresh_accelerator() -> Accelerator:
    Accelerator._instance = None
    return Accelerator()


def test_accelerator_manual_thread_override(monkeypatch):
    acc = _fresh_accelerator()
    monkeypatch.setattr(acc, "_system", "Darwin")
    monkeypatch.setattr(acc, "_machine", "arm64")
    monkeypatch.setattr(acc, "_num_threads", 10)

    observed = {"n": None}

    def _set_threads(n: int):
        observed["n"] = int(n)

    monkeypatch.setattr(cv2, "setNumThreads", _set_threads)
    acc.configure_runtime({"opencv_thread_count": 6})
    assert observed["n"] == 6


def test_accelerator_darwin_pcore_and_fallback(monkeypatch):
    acc = _fresh_accelerator()
    monkeypatch.setattr(acc, "_system", "Darwin")
    monkeypatch.setattr(acc, "_machine", "arm64")
    monkeypatch.setattr(acc, "_num_threads", 10)

    observed = {"n": None}

    def _set_threads(n: int):
        observed["n"] = int(n)

    monkeypatch.setattr(cv2, "setNumThreads", _set_threads)
    monkeypatch.setattr(acc, "_detect_macos_performance_cores", lambda: 8)
    acc.configure_runtime({"opencv_thread_count": 0})
    assert observed["n"] == 8

    monkeypatch.setattr(acc, "_detect_macos_performance_cores", lambda: None)
    acc.configure_runtime({"opencv_thread_count": 0})
    assert observed["n"] == 8  # round(10 * 0.75)
