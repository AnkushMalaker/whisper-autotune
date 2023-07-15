import json
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline

from whisper_asr.configs import Models
from whisper_asr.datatypes import PathLike

# with fsspec.open('s3://my-bucket/my-file.txt') as f:
#     contents = f.read()
#     print(contents)


def infer_file():
    # TODO: Add support for S3 using fsspec
    parser = ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?", default=None)
    parser.add_argument("--model", type=lambda x: Models[x], default=Models.WHISPER_TINY.name)

    args = parser.parse_args()
    # instantiate pipeline with bfloat16 and enable batching
    pipeline = FlaxWhisperPipline(
        args.model.value,
        dtype=jnp.bfloat16,
        batch_size=16,
    )

    # transcribe and return timestamps
    outputs = pipeline(str(args.input), task="transcribe", return_timestamps=True)
    if args.output is None:
        print(outputs)
    else:
        with open(args.output, "w") as f:
            f.write(json.dumps(outputs))

    # transcribe and return timestamps
    # outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)


if __name__ == "__main__":
    infer_file()
