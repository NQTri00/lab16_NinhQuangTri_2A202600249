#!/bin/bash
set -e

echo "Starting vLLM deployment (CPU mode)..."

# Install Docker (if not already present)
if ! command -v docker &> /dev/null; then
  apt-get update -y
  apt-get install -y docker.io
  systemctl enable docker
  systemctl start docker
fi

# Run vLLM in CPU mode
# Note: vLLM support for CPU is experimental in some versions. 
# We'll remove --gpus and --dtype half (which requires CUDA).
docker run -d \
  --restart unless-stopped \
  -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN="${hf_token}" \
  vllm/vllm-openai:latest \
  --model "${model_id}" \
  --device cpu \
  --max-model-len 2048
