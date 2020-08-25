#!/usr/bin/env bash

docker run --rm -it --init --gpus all --ipc=host --network=host -v ~/repos/lydorn/Polygonization-by-Frame-Field-Learning:/app -v ~/data:/data -e NVIDIA_VISIBLE_DEVICES=0 lydorn/frame-field-learning:1.4