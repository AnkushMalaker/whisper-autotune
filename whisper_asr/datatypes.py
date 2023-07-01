import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


@dataclass
class AudioFileCaptionPair:
    audio_file_path: PathLike
    caption: str

    def to_dict(self):
        return {"audio_file": str(self.audio_file_path), "caption": self.caption}

    def clean(self):
        self.caption = re.sub(r"[^a-zA-Z0-9 ]+", "", self.caption.lower())

    @classmethod
    def from_dict(cls, d):
        return cls(d["audio_file"], d["caption"])


@dataclass
class TimeStampCaptionPair:
    start_time: float
    end_time: float
    caption: str
