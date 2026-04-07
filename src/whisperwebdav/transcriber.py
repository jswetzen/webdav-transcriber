from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import Config


@dataclass
class TranscriptionResult:
    segments: list
    alignments_dir: Path
    workspace: Path


def transcribe(audio_local_path: str, config: Config) -> TranscriptionResult:
    """Transcribe an audio file using the easytranscriber pipeline.

    Creates a temporary workspace directory with vad/, transcriptions/,
    emissions/, and alignments/ subdirectories.

    The caller is responsible for cleaning up result.workspace.
    """
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

        audio_path = Path(audio_local_path)

        result = pipeline(
            config.vad_model,
            config.emissions_model,
            config.transcription_model,
            audio_paths=[audio_path.name],
            audio_dir=str(audio_path.parent),
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

        # alignment_pipeline returns list[list[Segment]] (one inner list per file)
        # We always pass one file, so unpack the first element
        segments = result[0] if result else []

        return TranscriptionResult(
            segments=segments,
            alignments_dir=alignments_dir,
            workspace=workspace,
        )
    except Exception:
        shutil.rmtree(workspace, ignore_errors=True)
        raise
