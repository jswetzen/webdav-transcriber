# TTS model alternatives (investigation, 2026-09-27)

A mostly academic look at whether to replace Kokoro-82M behind `/v1/audio/speech`,
prompted by Alibaba's Qwen-Audio-3.1-TTS blog post (September 2026). **Conclusion:
Kokoro stays.** Nothing below is urgent. This note records what was checked and what
a switch would involve, so the question doesn't have to be researched again from
scratch. Figures come from vendor pages and third-party write-ups as of the date
above; they were **not** benchmarked on our hardware.

## Qwen-Audio-3.1-TTS vs. Qwen3-TTS: two lineages

- **Qwen-Audio-3.x-TTS** (3.0 and the 3.1 in the blog) is **hosted-API only**, served
  through Alibaba Cloud Model Studio and resold via OpenRouter and others. No weights are
  published. Alibaba moved its flagship TTS from open to closed with 3.0. Version 3.1
  claims top scores on SEED-TTS-Eval, CV3-Eval and the Artificial Analysis TTS
  leaderboard; 16 languages plus 20 Chinese dialect regions; zero-shot cloning that holds
  up with noisy references; a 12.5 Hz speech tokenizer; up to 3 minutes of audio in one
  pass; and price cuts of roughly 70% against 3.0's ~$27.60 per million characters.
  **Swedish is not supported.** Using it would also mean sending every request's text to
  a third-party API, which defeats the point of a self-hosted service. Not considered
  further.
- **Qwen3-TTS** ([QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)) is the
  **open-weights** family, Apache 2.0, released January 2026. It comes in 0.6B and 1.7B
  sizes, each as Base (voice cloning), CustomVoice (preset voices) and VoiceDesign
  (describe a voice in text). It covers 10 languages: zh, en, ja, ko, de, fr, ru, pt, es
  and it. **Swedish is not supported.** Published streaming numbers at concurrency 1 are
  ~97 ms / ~101 ms to first audio and a real-time factor (RTF) of 0.288 / 0.313 for
  0.6B / 1.7B. vLLM-Omni has supported it since release day.

## Candidates

| Model | Size / license | Swedish | Speed | What it would add |
|---|---|---|---|---|
| **Kokoro-82M** (current) | 82M, Apache 2.0 | No (8 languages) | Much faster than real time; runs on CPU | Already integrated, streaming included |
| **Qwen3-TTS** 0.6B / 1.7B | Apache 2.0 | No | ~100 ms to first audio, RTF ~0.3 | Voice cloning, voice design from text, more expressive speech |
| **Chatterbox Multilingual** ([resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)) | 0.5B, MIT | **Yes** (23 languages) | Slower than Kokoro. The Turbo variant (350M, ~75 ms) is English only. | **Swedish**, zero-shot cloning, emotion control. Output is watermarked. |
| **Piper** (sv_SE voices) | Tiny, MIT | Yes | Instant on CPU | Swedish at almost no cost, but it sounds noticeably robotic |
| Orpheus / Higgs Audio V2 / Fish S2 Pro | 3B / ~5.8B / paid commercial license | Mostly English | Heavy | Nothing that justifies the cost here |

## What a switch would involve in this repo

- `server.py` only depends on four functions in `tts.py`: `synthesize()`,
  `synthesize_stream()`, `available_voices()` and `kokoro_files_present()`. A new engine
  means a second backend behind those four, probably chosen with a `TTS_ENGINE=kokoro|…`
  config variable. The OpenAI request format, the streaming protocol (including
  `pcm` → streaming for the customtts extension) and `GPU_LOCK` serialization all stay
  as they are.
- Kokoro runs on ONNX Runtime. Qwen3-TTS and Chatterbox are PyTorch models, so they
  need PyTorch in the image. The alternative is running them as a separate sidecar (for
  Qwen3-TTS, vLLM-Omni) and proxying to it. A sidecar sits outside the in-process
  `GPU_LOCK`, so TTS could run on the card at the same time as Whisper.
- VRAM: the `dev` box has an RTX 4070 Ti with 12 GB. Kokoro needs about 300 MB. A
  0.5–1.7B model needs a few GB and stays loaded next to KB-Whisper large and wav2vec2.
  It should fit, but check it first under a real transcription load.
- Voice-cloning models also need reference audio. That means a directory of reference
  clips to act as the "voice bank" behind `voice=` names.

## Recommendation

- **English only:** keep Kokoro. It is fast, small and already streams, and the
  alternatives mostly add cloning and expressiveness at a cost in VRAM, dependencies
  and time to first audio.
- **If Swedish speech is ever wanted** (a natural fit next to KB-Whisper), start with
  **Chatterbox Multilingual** as a second backend, keeping Kokoro for English. Piper is
  the cheap fallback if quality doesn't matter much.
- **Qwen3-TTS** is only worth adding for voice cloning or voice design. It does not
  help with Swedish.

## Sources

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS) ·
  [Qwen3-TTS open-source blog](https://qwen.ai/blog?id=qwen3tts-0115) ·
  [MarkTechPost on Qwen3-TTS](https://www.marktechpost.com/2026/01/22/qwen-researchers-release-qwen3-tts-an-open-multilingual-tts-suite-with-real-time-latency-and-fine-grained-voice-control/)
- [AlphaSignal: Qwen-Audio 3.1 pricing](https://alphasignal.ai/news/alibaba-s-qwen-audio-3-1-slashes-voice-api-prices-by-up-to-95) ·
  [OpenRouter: Qwen-Audio-3.0-TTS Plus](https://openrouter.ai/qwen/qwen-audio-3.0-tts-plus)
- [Chatterbox Multilingual (Resemble AI)](https://www.resemble.ai/learn/models/chatterbox-multilingual) ·
  [Chatterbox-TTS-Server (OpenAI-compatible)](https://github.com/devnen/Chatterbox-TTS-Server)
- [BentoML: open-source TTS 2026](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models) ·
  [SpeakEasy: open-source TTS 2026](https://www.tryspeakeasy.io/blog/open-source-text-to-speech-2026)
