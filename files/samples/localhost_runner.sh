#!/bin/bash

# Runs Piranha clients locally on 2 different GPUs
CUDA_VISIBLE_DEVICES=1,2 ./piranha -p 1 -c files/samples/localhost_config.json -v WARNING >/dev/null &
CUDA_VISIBLE_DEVICES=3,4 ./piranha -p 0 -c files/samples/localhost_config.json -v WARNING

