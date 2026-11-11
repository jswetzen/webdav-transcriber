"""One transcription job, run as a child process by isolated.py. Argv: AUDIO RESULT LANGUAGE.

Results go to RESULT as JSON, not stdout, because the logger writes to stdout.
"""

from __future__ import annotations

import json
import sys

from .config import Config
from .engine import transcribe_one


def main(argv: list[str]) -> int:
    audio_path, result_path, language = argv[1], argv[2], argv[3]
    config = Config()
    if language:
        config = config.model_copy(update={"language": language})
    segments = transcribe_one(audio_path, config)
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(segments, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
