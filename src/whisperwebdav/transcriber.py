from __future__ import annotations

import gc
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import structlog

from .config import Config

log = structlog.get_logger(__name__)


@dataclass
class BatchTranscriptionResult:
    segments_by_path: dict[str, list]
    workspace: Path


def release_gpu_memory() -> None:
    """Drop dead Python refs and return cached CUDA blocks to the driver.

    PyTorch's caching allocator holds freed VRAM in reserved blocks that other
    processes can't allocate. Call this between batches and on idle ticks so a
    co-located process (e.g. an LLM) can claim the GPU when we're not working.
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _is_empty_vad_indexerror(exc: BaseException) -> bool:
    """True if `exc` is easyaligner's known crash on a file with zero detected speech.

    easyaligner 0.2.3's Silero VAD wrapper (easyaligner/vad/silero.py, merge_chunks and
    run_vad_pipeline) indexes into the VAD segment list unconditionally (`segments[0]`,
    `segments[-1]`) with no empty-list guard. get_speech_timestamps() legitimately returns
    `[]` for audio with no detectable speech (pure music, long silence, near-inaudible
    recordings) -- not a bug in the recording, just nothing to align -- and that crashes
    the whole pipeline() call with a bare IndexError instead of a "0 segments" result.
    Confirmed by hand 2026-09-12: reproduced against the server directly for the four
    files that were 500ing and looping ~1000x/day each in production, isolated with a full
    traceback to exactly these two lines. Matched by frame filename rather than message so
    an unrelated IndexError elsewhere in the pipeline still propagates normally.
    """
    if not isinstance(exc, IndexError):
        return False
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename.endswith("easyaligner/vad/silero.py"):
            return True
        tb = tb.tb_next
    return False


def transcribe_batch(
    audio_local_paths: list[str], config: Config
) -> BatchTranscriptionResult:
    """Transcribe a batch of audio files in a single easytranscriber pipeline call.

    All paths must reside in the same parent directory (easytranscriber's pipeline
    takes one audio_dir + a list of filenames). The caller is responsible for
    cleaning up result.workspace.
    """
    if not audio_local_paths:
        raise ValueError("audio_local_paths must be non-empty")

    paths = [Path(p) for p in audio_local_paths]
    parents = {p.parent for p in paths}
    if len(parents) != 1:
        raise ValueError(
            f"All audio paths must share one parent directory; got {parents}"
        )
    audio_dir = next(iter(parents))

    # Import here to avoid slow startup if module not available
    from easyaligner.text import load_tokenizer, text_normalizer  # type: ignore[import]
    from easytranscriber.pipelines import pipeline  # type: ignore[import]

    workspace = Path(tempfile.mkdtemp())
    try:
        vad_dir = workspace / "vad"
        transcriptions_dir = workspace / "transcriptions"
        emissions_dir = workspace / "emissions"
        alignments_dir = workspace / "alignments"

        for d in (vad_dir, transcriptions_dir, emissions_dir, alignments_dir):
            d.mkdir()

        try:
            result = pipeline(
                config.vad_model,
                config.emissions_model,
                config.transcription_model,
                audio_paths=[p.name for p in paths],
                audio_dir=str(audio_dir),
                language=config.language,
                tokenizer=load_tokenizer(config.tokenizer_name),
                text_normalizer_fn=text_normalizer,
                cache_dir=config.cache_dir,
                output_vad_dir=str(vad_dir),
                output_transcriptions_dir=str(transcriptions_dir),
                output_emissions_dir=str(emissions_dir),
                output_alignments_dir=str(alignments_dir),
                save_json=True,
                return_alignments=True,
                device=config.device,
                hf_token=config.hf_token or None,
            )
        except IndexError as exc:
            # Batch size is always 1 in this deployment (the server is only ever called
            # per-file, via the poller's TRANSCRIBE_BACKEND=http thin client), so a
            # VAD-empty crash unambiguously means *this* file. With more than one file we
            # can't attribute a batch-wide crash to a specific one without re-running per
            # file, so keep the existing all-fail-together behavior for that case.
            if len(audio_local_paths) == 1 and _is_empty_vad_indexerror(exc):
                log.warning(
                    "No speech detected by VAD; returning an empty transcript instead of "
                    "failing (known easyaligner gap, see _is_empty_vad_indexerror)",
                    filename=paths[0].name,
                )
                return BatchTranscriptionResult(
                    segments_by_path={audio_local_paths[0]: []},
                    workspace=workspace,
                )
            raise

        # alignment_pipeline returns list[list[Segment]] — one inner list per file,
        # in the same order as audio_paths.
        segments_by_path: dict[str, list] = {}
        for i, local_path in enumerate(audio_local_paths):
            segments_by_path[local_path] = result[i] if i < len(result) else []

        return BatchTranscriptionResult(
            segments_by_path=segments_by_path,
            workspace=workspace,
        )
    except Exception:
        shutil.rmtree(workspace, ignore_errors=True)
        raise
    finally:
        release_gpu_memory()
