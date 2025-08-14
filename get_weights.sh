#!/bin/bash

set -x

WEIGHT_FILE="gpt2-small-124M.safetensors"
# WEIGHT_FILE="gpt2-medium-355M.safetensors"
# WEIGHT_FILE="gpt2-large-774M.safetensors"
# WEIGHT_FILE="gpt2-xl-1558M.safetensors"

URL="https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/$WEIGHT_FILE"

curl -o $WEIGHT_FILE -L $URL
