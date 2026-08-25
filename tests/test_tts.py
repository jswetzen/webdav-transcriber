from __future__ import annotations

import pytest

from whisperwebdav.config import Config
from whisperwebdav import tts


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch):
    # tts.py caches the Kokoro instance/voices at module scope; isolate tests from each other.
    monkeypatch.setattr(tts, "_kokoro", None)
    monkeypatch.setattr(tts, "_voices_cache", None)


class FakeKokoro:
    def __init__(self):
        self.calls = []

    def get_voices(self):
        return ["af_sarah", "af_sky"]

    def create(self, text, *, voice, speed, lang):
        self.calls.append((text, voice, speed, lang))
        return [0.0] * 16000, 16000


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
