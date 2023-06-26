# Code for whisper-asr (autotuning)

## Usage
### 1. Docker (Recommended)
1. Build and start the docker image with `docker-compose up development --build`
2. To use the container, run `docker exec -it whisper-asr-development bash`
3. Now you can run the script for ASR like so: `whisper-asr /path/to/audio.wav /optional/path/to/output.json --model <optional-model-name>`. If no `--model` is specified, the whisper-tiny model is used. If no output is provided, the output is printed to stdout.

## Development
### 1. Docker (Recommended)
1. Build and start the docker image with `docker-compose up development --build`
2. To use the container, run `docker exec -it whisper-asr-development bash`
3. Stop the container using `docker-compose down` or `docker stop whisper-asr-development`
4. To remove the container, run `docker rm whisper-asr-development`.

This mounts the current directory as a volume, meaning any files you put in the current directory will be available in the container. Similarly you files you delete within the container, will be deleted in the current directory.
This makes it easy to put audio files in the directory as you need, and use the container to run the code.
The Data directory is mounted at ~/whisper-data\
See docker-compose.yaml for more info.

### 2. Local
1. Install python3.10\
2. Install poetry
3. Use `poetry install` to install dependencies, followed by `sh ./install_non_poetry.sh` to install non-poetry dependencies (jax cuda version).
4. Run `poetry shell` to spawn a shell with the poetry environment activated.