@echo off
set model=TheBloke/zephyr-7B-beta-GPTQ
set volume=%~dp0../../data

docker run --gpus all --shm-size 1g -p 8080:80 -v %volume%:/data ghcr.io/huggingface/text-generation-inference:latest --model-id %model%
pause