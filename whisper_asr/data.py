import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import youtube_dl
from pydub import AudioSegment
from youtube_dl.utils import DownloadError

from whisper_asr.datatypes import PathLike

logger = logging.getLogger(__name__)


def my_hook(d):
    if d["status"] == "finished":
        print("Done downloading, now converting ...")


def download_data(video_urls: Optional[List[str]] = None) -> Dict[str, Dict[str, str]]:
    if video_urls is None:
        video_urls = [
            "https://www.youtube.com/watch?v=f4k6xRfmz2A",
            "https://www.youtube.com/watch?v=mPF9f-PLDPc",
            "https://www.youtube.com/watch?v=D24ueW8G0-w",
            "https://www.youtube.com/watch?v=dXqfEFX1veY",
            "https://www.youtube.com/watch?v=dc7CIkZcWYE",
            "https://www.youtube.com/watch?v=1vpepaQ-VQQ",
            "https://www.youtube.com/watch?v=nkh9VGCY8as",
        ]

    download_dir = Path("Data")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "subtitleslangs": ["en-US"],
        "subtitlesformat": "vtt",
        "writesubtitles": True,
        "logger": logger,
        "progress_hooks": [my_hook],
        "writedescriptions": True,
        "restrictfilenames": True,
        # 'ignoreerrors': True,
    }

    audio_dir = download_dir / "audio"
    subtitle_dir = download_dir / "subtitle"
    audio_dir.mkdir(exist_ok=True)
    subtitle_dir.mkdir(exist_ok=True)

    json_file = download_dir / "audio_subtitle_pairs.json"
    data_dict = {}

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(video_urls)
    pairs = sorted(
        [
            x
            for x in Path.cwd().iterdir()
            if not x.is_dir() and (x.suffix == ".vtt" or x.suffix == ".wav")
        ]
    )
    for i in range(0, len(pairs), 2):
        file1 = pairs[i]
        if file1.suffix == ".vtt":
            subtitle_file = file1
            audio_file = pairs[i + 1]
        else:
            audio_file = file1
            subtitle_file = pairs[i + 1]
        audio_file = audio_file.rename(audio_dir / audio_file.name)
        subtitle_file = subtitle_file.rename(subtitle_dir / subtitle_file.name)
        audio_name = audio_file.stem
        subtitle_name = subtitle_file.stem
        # assert audio_name == subtitle_name
        data_dict[audio_name] = {
            "audio": str(audio_file.resolve()),
            "subtitle": str(subtitle_file.resolve()),
        }

    with open(json_file, "w") as f:
        json.dump(data_dict, f)

    return data_dict


@dataclass
class AudioFileCaptionPair:
    audio_file: PathLike
    caption: str

    def to_dict(self):
        return {"audio_file": str(self.audio_file), "caption": self.caption}

    def clean(self):
        self.caption = re.sub(r"[^a-zA-Z0-9 ]+", "", self.caption.lower())


@dataclass
class TimeStampCaptionPair:
    start_time: float
    end_time: float
    caption: str


def convert_timestamp_to_seconds(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    seconds, milliseconds = seconds.split(".")
    hours = float(hours)
    minutes = float(minutes)
    seconds = float(seconds)
    milliseconds = float(milliseconds)
    return hours * 60 * 60 + minutes * 60 + seconds + milliseconds / 1000


def parse_vtt(vtt_file: PathLike) -> List[TimeStampCaptionPair]:
    with open(vtt_file, "r") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    lines = lines[4:]

    i = 0
    time_stamp_caption_pairs = []
    while i < len(lines):
        start_time, end_time = lines[i].split("-->")

        start_time = convert_timestamp_to_seconds(start_time.strip())
        end_time = convert_timestamp_to_seconds(end_time.strip())
        caption_lines = []
        j = i
        while j < len(lines) and lines[j] != "":
            caption_lines.append(lines[j])
            j += 1
        caption = " ".join(caption_lines)
        time_stamp_caption_pairs.append(
            TimeStampCaptionPair(
                start_time=float(start_time), end_time=float(end_time), caption=caption
            )
        )
        i = j + 1

    return time_stamp_caption_pairs


def preprocess_data(
    data_dict: Dict[str, Dict[str, str]], output_dir: PathLike, output_json: PathLike
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    audio_caption_pairs: List[AudioFileCaptionPair] = []
    for data_name, data in data_dict.items():
        audio_file = Path(data["audio"])
        subtitle_file = Path(data["subtitle"])
        time_stamp_caption_pairs = parse_vtt(subtitle_file)

        audio = AudioSegment.from_wav(audio_file)
        list_of_timestamps = [
            (x.start_time, x.end_time) for x in time_stamp_caption_pairs
        ]

        for idx, t in enumerate(list_of_timestamps):
            # break loop if at last element of list
            if idx == len(list_of_timestamps):
                break
            start = t[0] * 1000
            end = (
                list_of_timestamps[idx][1] * 1000
                if idx + 1 < len(list_of_timestamps)
                else len(audio)
            )
            segment = audio[start:end]
            output_file = output_dir / f"{data_name}_{idx}.wav"
            segment.export(output_file, format="wav")

            audio_caption_pairs.append(
                AudioFileCaptionPair(output_file, time_stamp_caption_pairs[idx].caption)
            )

        with open(output_json, "w") as f:
            json.dump([acp.to_dict() for acp in audio_caption_pairs], f)


def preprocess_caption(caption: str) -> str:
    new_string = re.sub(r"\([^)]*\)", "", caption)  # remove text within parentheses

    return new_string


if __name__ == "__main__":
    if Path("Data/audio_subtitle_pairs.json").exists():
        with open("Data/audio_subtitle_pairs.json", "r") as f:
            data_dict = json.load(f)
    else:
        data_dict = download_data()

    preprocess_data(
        data_dict=data_dict,
        output_dir="Data/preprocessed",
        output_json="Data/audio_caption_pairs.json",
    )
