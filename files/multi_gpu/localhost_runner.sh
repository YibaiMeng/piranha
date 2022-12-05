#!/bin/bash

# Runs Piranha clients locally on 3 different GPUs
CUDA_VISIBLE_DEVICES=3 ./piranha -p 1 -c files/samples/localhost_config.json >/dev/null &
CUDA_VISIBLE_DEVICES=4 ./piranha -p 2 -c files/samples/localhost_config.json >/dev/null &
CUDA_VISIBLE_DEVICES=5 ./piranha -p 0 -c files/samples/localhost_config.json

