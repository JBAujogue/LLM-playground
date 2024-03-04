#!/usr/bin/env sh
model=TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
volume="$PWD/data/tgi"

docker run --gpus all --shm-size 1g -p 8080:80 -v "$volume":/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model --quantize gptq
