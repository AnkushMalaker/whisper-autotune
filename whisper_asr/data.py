import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchaudio
import youtube_dl
from datasets import Dataset
from pydub import AudioSegment
from transformers.models.whisper import (
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
)

from whisper_asr.datatypes import AudioFileCaptionPair, PathLike, TimeStampCaptionPair

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/root/whisper-data"))


def prepare_sample(batch, feature_extractor, tokenizer, resampling_rate=16000):
    audio = batch["audio"]
    audio["array"] = torchaudio.functional.resample(
        waveform=torch.tensor(audio["array"]),
        orig_freq=audio["sampling_rate"],
        new_freq=resampling_rate,
    )
    audio["sampling_rate"] = resampling_rate
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    )["input_features"][0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids

    return batch


def prepare_data(
    audio_caption_pairs_list: List[AudioFileCaptionPair],
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    target_sampling_rate: int = 16000,
    dataset_path: Optional[PathLike] = None,
    force_recompute: bool = False,
):
    """
    The data is in the form of audio-caption pairs.
    {'audio': {'path': '/home/sanchit_huggingface_co/.cache/huggingface/datasets/downloads/extracted/607848c7e74a89a3b5225c0fa5ffb9470e39b7f11112db614962076a847f3abf/cv-corpus-11.0-2022-09-21/hi/clips/common_voice_hi_25998259.mp3',
            'array': array([0.0000000e+00, 0.0000000e+00, 0.0000000e+00, ..., 9.6724887e-07,
        1.5334779e-06, 1.0415988e-06], dtype=float32),
            'sampling_rate': 48000},
    'sentence': 'खीर की मिठास पर गरमाई बिहार की सियासत, कुशवाहा ने दी सफाई'}
    """
    if dataset_path and Path(dataset_path).exists() and not force_recompute:
        audio_dataset = Dataset.load_from_disk(str(dataset_path))
        return audio_dataset

    data_list = []

    for audio_caption_pair in audio_caption_pairs_list:
        audio_waveform: torch.Tensor
        audio_waveform, sample_rate = torchaudio.load(  # type: ignore
            audio_caption_pair.audio_file_path
        )
        feature_dict = {
            "audio": {
                "path": audio_caption_pair.audio_file_path,
                "array": audio_waveform.mean(0),
                "sampling_rate": sample_rate,
            },
            "sentence": audio_caption_pair.caption,
        }
        data_list.append(feature_dict)
    audio_dataset = Dataset.from_list(data_list)
    # audio_dataset = audio_dataset.cast_column(
    #     "audio", Audio(sampling_rate=target_sampling_rate)
    # )
    # audio_dataset = audio_dataset.map(

    audio_dataset = audio_dataset.map(
        prepare_sample,
        fn_kwargs={
            "feature_extractor": feature_extractor,
            "tokenizer": tokenizer,
            "resampling_rate": target_sampling_rate,
        },
        num_proc=1,
    )
    if dataset_path is not None:
        audio_dataset.save_to_disk(str(dataset_path))

    return audio_dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def my_hook(d):
    if d["status"] == "finished":
        print("Done downloading, now converting ...")


def download_data(video_urls: Optional[List[str]] = None) -> Dict[str, Dict[str, str]]:
    if video_urls is None:
        video_urls = list(
            set(
                [
                    "https://www.youtube.com/watch?v=f4k6xRfmz2A",
                    "https://www.youtube.com/watch?v=mPF9f-PLDPc",
                    "https://www.youtube.com/watch?v=D24ueW8G0-w",
                    "https://www.youtube.com/watch?v=dXqfEFX1veY",
                    "https://www.youtube.com/watch?v=dc7CIkZcWYE",
                    "https://www.youtube.com/watch?v=1vpepaQ-VQQ",
                    "https://www.youtube.com/watch?v=nkh9VGCY8as",
                    "https://www.youtube.com/watch?v=1ugJ1BJx0HE",
                    "https://www.youtube.com/watch?v=9CunwUs08og",
                    "https://www.youtube.com/watch?v=UeCdBVHYa_8",
                    "https://www.youtube.com/watch?v=IawfrWLDN4U",
                    "https://www.youtube.com/watch?v=QOhLlvNlI20",
                    "https://www.youtube.com/watch?v=Z07ZNsWGC80",
                    "https://www.youtube.com/watch?v=ykDuoq-MpHg",
                    "https://www.youtube.com/watch?v=RYMnIGxxqU0",
                    "https://www.youtube.com/watch?v=J2UaipfsR7Q",
                    "https://www.youtube.com/watch?v=TzntUW34bv8",
                    "https://www.youtube.com/watch?v=FPF7Z7TLdsk",
                    "https://www.youtube.com/watch?v=auObtDOftAI",
                    "https://www.youtube.com/watch?v=-PD0FZt9-VU",
                    "https://www.youtube.com/watch?v=GgK1o5ytXr8",
                    "https://www.youtube.com/watch?v=jgzI-N_U2hs",
                    "https://www.youtube.com/watch?v=kuTTAuUorsI",
                    "https://www.youtube.com/watch?v=S0zpKHvEKXc",
                    "https://www.youtube.com/watch?v=RFzirpvTiOo",
                ]
            )
        )

    download_dir = DATA_DIR
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
    i = 0
    while i < len(pairs):
        file1 = pairs[i]
        if file1.suffix == ".vtt":
            subtitle_file = file1
            audio_file = pairs[i + 1]
        else:
            audio_file = file1
            subtitle_file = pairs[i + 1]
        if audio_file.name[0:15] != subtitle_file.name[0:15]:
            print(f"Skipping {audio_file.name} as it doesn't match {subtitle_file.name}")
            if audio_file.exists():
                audio_file.unlink()
            else:
                subtitle_file.unlink()
            i += 1
            continue
        audio_file = shutil.move(audio_file, audio_dir / audio_file.name)
        subtitle_file = shutil.move(subtitle_file, subtitle_dir / subtitle_file.name)
        audio_name = audio_file.stem
        subtitle_name = subtitle_file.stem
        # assert audio_name == subtitle_name
        data_dict[audio_name] = {
            "audio": str(audio_file.resolve()),
            "subtitle": str(subtitle_file.resolve()),
        }
        i += 2

    with open(json_file, "w") as f:
        json.dump(data_dict, f)

    return data_dict


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
        i += 1

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
        list_of_timestamps = [(x.start_time, x.end_time) for x in time_stamp_caption_pairs]

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

            txt = preprocess_caption(time_stamp_caption_pairs[idx].caption)
            if txt == "" or txt == " ":
                continue
            audio_caption_pairs.append(
                AudioFileCaptionPair(
                    output_file,
                    preprocess_caption(time_stamp_caption_pairs[idx].caption),
                )
            )

        with open(output_json, "w") as f:
            json.dump([acp.to_dict() for acp in audio_caption_pairs], f)


def preprocess_caption(caption: str) -> str:
    new_string = re.sub(r"\([^)]*\)", "", caption)  # remove text within parentheses
    new_string = re.sub(r"\[[^)]*\]", "", new_string)  # remove text within square brackets
    new_string = new_string.lower().replace("-", "").strip()
    # TODO: Add hyphens etc from language model
    if new_string == "":
        return " "
    return new_string


if __name__ == "__main__":
    if (DATA_DIR / "audio_subtitle_pairs.json").exists():
        with open(DATA_DIR / "audio_subtitle_pairs.json", "r") as f:
            data_dict = json.load(f)
    else:
        data_dict = download_data()

    preprocess_data(
        data_dict=data_dict,
        output_dir=DATA_DIR / "preprocessed",
        output_json=DATA_DIR / "audio_caption_pairs.json",
    )
