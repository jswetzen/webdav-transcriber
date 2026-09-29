from __future__ import annotations

import asyncio
import shutil

import pytest

from whisperwebdav.config import Config
from whisperwebdav import tts

# The real webdav-transcriber image always has ffmpeg (Dockerfile installs it for Whisper's
# input demuxing too, see tts.py's _wav_to_mp3 docstring), but a bare CI/dev Python env may
# not -- skip rather than fail so `pytest` still passes without it installed.
requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _model_files(tmp_path):
    model = tmp_path / "kokoro-v1.0.onnx"
    voices = tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    return model, voices


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch):
    # tts.py caches the Kokoro instance/voices at module scope; isolate tests from each other.
    monkeypatch.setattr(tts, "_kokoro", None)
    monkeypatch.setattr(tts, "_voices_cache", None)
    monkeypatch.setattr(tts, "_watchdog", None)
    monkeypatch.setattr(tts, "_last_used", tts.time.monotonic())


class FakeKokoro:
    def __init__(self, stream_chunks=None):
        self.calls = []
        self.stream_calls = []
        # (samples, sample_rate) tuples create_stream() yields; defaults to two small chunks
        # so tests exercise the multi-chunk / eager-drain path in _stream_pcm_to_mp3.
        self._stream_chunks = stream_chunks if stream_chunks is not None else [
            ([0.0] * 4000, 16000),
            ([0.0] * 4000, 16000),
        ]

    def get_voices(self):
        return ["af_sarah", "af_sky"]

    def create(self, text, *, voice, speed, lang):
        self.calls.append((text, voice, speed, lang))
        return [0.0] * 16000, 16000

    async def create_stream(self, text, *, voice, speed, lang):
        self.stream_calls.append((text, voice, speed, lang))
        for chunk in self._stream_chunks:
            yield chunk


def test_kokoro_files_present_false_by_default() -> None:
    assert tts.kokoro_files_present(Config()) is False


def test_synthesize_missing_model_files_raises() -> None:
    with pytest.raises(tts.TTSError, match="not found"):
        tts.synthesize("hello", Config())


def test_synthesize_unsupported_format_raises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    model = tmp_path / "kokoro-v1.0.onnx"
    voices = tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    config = Config(kokoro_model_path=str(model), kokoro_voices_path=str(voices))

    monkeypatch.setattr(tts, "_load_kokoro", lambda cfg: FakeKokoro())

    with pytest.raises(tts.TTSError, match="Unsupported response_format"):
        tts.synthesize("hi", config, response_format="ogg")


def test_synthesize_unknown_voice_raises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    model = tmp_path / "kokoro-v1.0.onnx"
    voices = tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    config = Config(kokoro_model_path=str(model), kokoro_voices_path=str(voices))

    monkeypatch.setattr(tts, "_load_kokoro", lambda cfg: FakeKokoro())

    with pytest.raises(tts.TTSError, match="Unknown voice"):
        tts.synthesize("hi", config, voice="bogus")


def test_synthesize_wav_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    model = tmp_path / "kokoro-v1.0.onnx"
    voices = tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    config = Config(kokoro_model_path=str(model), kokoro_voices_path=str(voices))

    fake = FakeKokoro()
    monkeypatch.setattr(tts, "_load_kokoro", lambda cfg: fake)

    result = tts.synthesize("hi there", config, voice="af_sky", response_format="wav")

    assert result.media_type == "audio/wav"
    assert result.audio.startswith(b"RIFF")
    assert result.duration_seconds == pytest.approx(1.0)
    assert fake.calls == [("hi there", "af_sky", 1.0, config.tts_lang)]


def test_synthesize_defaults_to_configured_voice(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    model = tmp_path / "kokoro-v1.0.onnx"
    voices = tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")
    config = Config(
        kokoro_model_path=str(model),
        kokoro_voices_path=str(voices),
        tts_default_voice="af_sarah",
    )

    fake = FakeKokoro()
    monkeypatch.setattr(tts, "_load_kokoro", lambda cfg: fake)

    tts.synthesize("hi", config)

    assert fake.calls[0][1] == "af_sarah"


# --- synthesize_stream ---
#
# No async test runner (pytest-asyncio/anyio-pytest) is a dev dependency, so these stay plain
# sync tests and drive the coroutines with asyncio.run() directly rather than adding one.


async def _drain(agen):
    return b"".join([chunk async for chunk in agen])


def test_synthesize_stream_rejects_wav(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    model, voices = _model_files(tmp_path)
    config = Config(kokoro_model_path=str(model), kokoro_voices_path=str(voices))

    fake = FakeKokoro()
    monkeypatch.setattr(tts, "_load_kokoro", lambda cfg: fake)

    with pytest.raises(tts.TTSError, match="only supports response_format='mp3'"):
        asyncio.run(tts.synthesize_stream("hi", config, response_format="wav").__anext__())

    # Rejected before touching kokoro at all.
    assert fake.stream_calls == []
    # GPU_LOCK.acquire() never even runs for this branch, so it must still be free.
    assert tts.GPU_LOCK.acquire(blocking=False)
    tts.GPU_LOCK.release()


def test_synthesize_stream_unknown_voice_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model, voices = _model_files(tmp_path)
    config = Config(kokoro_model_path=str(model), kokoro_voices_path=str(voices))

    fake = FakeKokoro()
    monkeypatch.setattr(tts, "_load_kokoro", lambda cfg: fake)

    with pytest.raises(tts.TTSError, match="Unknown voice"):
        asyncio.run(tts.synthesize_stream("hi", config, voice="bogus").__anext__())

    # GPU_LOCK was acquired-then-released around the failed validation, not leaked.
    assert tts.GPU_LOCK.acquire(blocking=False)
    tts.GPU_LOCK.release()


@requires_ffmpeg
def test_synthesize_stream_yields_mp3_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    model, voices = _model_files(tmp_path)
    config = Config(kokoro_model_path=str(model), kokoro_voices_path=str(voices))

    fake = FakeKokoro()
    monkeypatch.setattr(tts, "_load_kokoro", lambda cfg: fake)

    audio = asyncio.run(_drain(tts.synthesize_stream("hi there", config, voice="af_sky")))

    assert audio.startswith(b"ID3") or audio[0:1] == b"\xff"  # mp3 tag or raw frame sync
    assert fake.stream_calls == [("hi there", "af_sky", 1.0, config.tts_lang)]
    # GPU_LOCK released once the generator is exhausted.
    assert tts.GPU_LOCK.acquire(blocking=False)
    tts.GPU_LOCK.release()


def test_synthesize_stream_empty_chunks_yields_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model, voices = _model_files(tmp_path)
    config = Config(kokoro_model_path=str(model), kokoro_voices_path=str(voices))

    fake = FakeKokoro(stream_chunks=[])
    monkeypatch.setattr(tts, "_load_kokoro", lambda cfg: fake)

    audio = asyncio.run(_drain(tts.synthesize_stream("hi", config)))

    assert audio == b""
    assert tts.GPU_LOCK.acquire(blocking=False)
    tts.GPU_LOCK.release()


def test_synthesize_stream_pcm_yields_raw_s16le(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # Two chunks of known float32 samples; no ffmpeg needed for pcm at all, so this runs
    # everywhere -- unlike the mp3 path, verifiable exactly rather than just "looks like mp3".
    model, voices = _model_files(tmp_path)
    config = Config(kokoro_model_path=str(model), kokoro_voices_path=str(voices))

    fake = FakeKokoro(stream_chunks=[([0.0, 0.5, -1.0, 1.0], 24000), ([-0.5], 24000)])
    monkeypatch.setattr(tts, "_load_kokoro", lambda cfg: fake)

    audio = asyncio.run(
        _drain(tts.synthesize_stream("hi there", config, voice="af_sky", response_format="pcm"))
    )

    import numpy as np

    samples = np.frombuffer(audio, dtype="<i2")
    # 32767, not 32768, so exact equality on the +1.0 sample would be off by one; approx covers
    # the intentional clamp-to-int16-range rounding for every value here.
    np.testing.assert_allclose(
        samples / 32767.0, [0.0, 0.5, -1.0, 1.0, -0.5], atol=1e-4
    )
    assert fake.stream_calls == [("hi there", "af_sky", 1.0, config.tts_lang)]
    assert tts.GPU_LOCK.acquire(blocking=False)
    tts.GPU_LOCK.release()


def test_synthesize_stream_rejects_bad_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model, voices = _model_files(tmp_path)
    config = Config(kokoro_model_path=str(model), kokoro_voices_path=str(voices))

    fake = FakeKokoro()
    monkeypatch.setattr(tts, "_load_kokoro", lambda cfg: fake)

    with pytest.raises(tts.TTSError, match="only supports response_format='mp3' or 'pcm'"):
        asyncio.run(tts.synthesize_stream("hi", config, response_format="ogg").__anext__())

    assert fake.stream_calls == []


# --- Idle release (tts_idle_release_seconds) -------------------------------------------------
# These drive _release_if_idle() directly rather than waiting on the watchdog thread, so they
# don't depend on real sleeps.


def test_release_if_idle_drops_model_after_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts, "_kokoro", FakeKokoro())
    monkeypatch.setattr(tts, "_last_used", tts.time.monotonic() - 1000)

    assert tts._release_if_idle(300) is True
    assert tts._kokoro is None


def test_release_if_idle_waits_while_gpu_lock_held(monkeypatch: pytest.MonkeyPatch) -> None:
    # A synthesis or stream in flight holds GPU_LOCK; the watchdog must back off, not block.
    monkeypatch.setattr(tts, "_kokoro", FakeKokoro())
    monkeypatch.setattr(tts, "_last_used", tts.time.monotonic() - 1000)

    tts.GPU_LOCK.acquire()
    try:
        assert tts._release_if_idle(300) is False
        assert tts._kokoro is not None
    finally:
        tts.GPU_LOCK.release()

    assert tts._release_if_idle(300) is True
    assert tts._kokoro is None


def test_release_if_idle_keeps_model_before_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeKokoro()
    monkeypatch.setattr(tts, "_kokoro", fake)
    monkeypatch.setattr(tts, "_last_used", tts.time.monotonic() - 10)

    assert tts._release_if_idle(300) is False
    assert tts._kokoro is fake


def _install_fake_kokoro_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    module = types.ModuleType("kokoro_onnx")
    module.Kokoro = lambda model_path, voices_path: FakeKokoro()
    monkeypatch.setitem(sys.modules, "kokoro_onnx", module)


def test_idle_release_zero_starts_no_watchdog(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _install_fake_kokoro_module(monkeypatch)
    model, voices = _model_files(tmp_path)
    config = Config(
        kokoro_model_path=str(model), kokoro_voices_path=str(voices), tts_idle_release_seconds=0
    )

    assert isinstance(tts._load_kokoro(config), FakeKokoro)
    assert tts._watchdog is None


def test_load_starts_watchdog_that_releases(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _install_fake_kokoro_module(monkeypatch)
    monkeypatch.setattr(tts, "_WATCHDOG_MAX_POLL_SECONDS", 0.01)
    model, voices = _model_files(tmp_path)
    config = Config(
        kokoro_model_path=str(model), kokoro_voices_path=str(voices), tts_idle_release_seconds=1
    )

    tts._load_kokoro(config)
    watchdog = tts._watchdog
    assert watchdog is not None and watchdog.name == "kokoro-idle-release"
    # Pretend the last request was long ago so the next poll releases.
    monkeypatch.setattr(tts, "_last_used", tts.time.monotonic() - 1000)
    watchdog.join(timeout=5)

    assert not watchdog.is_alive()
    assert tts._kokoro is None
    assert tts._watchdog is None
