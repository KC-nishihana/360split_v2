import cv2

from core.video_loader import create_video_capture


class _FakeCap:
    def __init__(self, opened: bool):
        self._opened = bool(opened)

    def isOpened(self):
        return self._opened

    def release(self):
        return None


def test_create_video_capture_darwin_auto_fallback(monkeypatch):
    calls = []

    def _fake_vc(_path, backend=None):
        calls.append(backend)
        if backend == cv2.CAP_AVFOUNDATION:
            return _FakeCap(False)
        if backend == cv2.CAP_FFMPEG:
            return _FakeCap(True)
        return _FakeCap(False)

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(cv2, "VideoCapture", _fake_vc)

    cap = create_video_capture("dummy.mp4", backend_preference="auto")
    assert cap.isOpened() is True
    assert calls[:2] == [cv2.CAP_AVFOUNDATION, cv2.CAP_FFMPEG]


def test_create_video_capture_darwin_ffmpeg_priority(monkeypatch):
    calls = []

    def _fake_vc(_path, backend=None):
        calls.append(backend)
        if backend == cv2.CAP_FFMPEG:
            return _FakeCap(True)
        return _FakeCap(False)

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(cv2, "VideoCapture", _fake_vc)

    cap = create_video_capture("dummy.mp4", backend_preference="ffmpeg")
    assert cap.isOpened() is True
    assert calls[0] == cv2.CAP_FFMPEG
