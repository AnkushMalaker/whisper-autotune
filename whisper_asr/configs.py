from enum import Enum


class Models(str, Enum):
    WHISPER_TINY = "openai/whisper-tiny"
    WHISPER_BASE = "openai/whisper-base"
    WHISPER_SMALL = "openai/whisper-small"
    WHISPER_MEDIUM = "openai/whisper-medium"
    WHISPER_LARGE = "openai/whisper-large"
    WHISPER_LARGE_V2 = "openai/whisper-large-v2"
    WHISPER_SMALL_HI = "sanchit-gandhi/whisper-small-hi"
