#!/bin/bash

set -x

curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "openai-community/gpt2", "messages": [ {"role": "user", "content": "Hello! Can you help me with a simple task?"} ], "max_tokens": 100 }'
