ARG POETRY_HOME="/opt/poetry"
ARG PYSETUP_PATH="/opt/code"
ARG VENV_PATH="/opt/code/.venv"

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as builder

ARG POETRY_HOME
ARG PYSETUP_PATH
ARG VENV_PATH

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME=$POETRY_HOME \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH=$PYSETUP_PATH \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_CREATE=false

ENV POETRY_VERSION=1.5.1

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


RUN apt-get update
RUN apt-get install --no-install-recommends -y \
        curl \
        build-essential
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install git ffmpeg libsm6 libxext6 vim -y

RUN curl -sSL https://install.python-poetry.org | python3 -

FROM builder as development

WORKDIR $PYSETUP_PATH
COPY ./poetry.lock ./pyproject.toml ./README.md ./
COPY ./whisper_asr/ ./whisper_asr

RUN poetry install
COPY ./install_non_poetry.sh .
RUN sh ./install_non_poetry.sh

ENV JAX_PLATFORM_NAME cuda
ENV DATA_DIR /root/whisper-data