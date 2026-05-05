from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import Config


@dataclass
class BatchTranscriptionResult:
    segments_by_path: dict[str, list]
    workspace: Path


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
