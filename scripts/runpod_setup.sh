#!/bin/bash
set -e

echo "=== [1/7] GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo "=== [2/7] Installing FFmpeg ==="
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
ffmpeg -version | head -1

echo "=== [3/7] Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
OLLAMA_PID=$!
sleep 5

echo "=== [4/7] Pulling Ollama models ==="
ollama pull llama3.2
ollama pull qwen2.5vl

echo "=== [5/7] Cloning ComfyUI ==="
cd /workspace
if [ ! -d "ComfyUI" ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
fi
cd ComfyUI
pip install -q -r requirements.txt

echo "=== [5b/7] Installing MiniMax H3 custom nodes ==="
cd /workspace/ComfyUI/custom_nodes
if [ ! -d "ComfyUI-Hailuo" ]; then
  git clone --depth 1 https://github.com/MiniMax-AI/ComfyUI-Hailuo.git
fi
if [ -f "ComfyUI-Hailuo/requirements.txt" ]; then
  pip install -q -r ComfyUI-Hailuo/requirements.txt 2>/dev/null || true
fi

echo "=== [6/7] Cloning H3VideoGen ==="
cd /workspace
if [ ! -d "H3VideoGen" ]; then
  git clone --depth 1 https://github.com/placeholder/H3VideoGen.git 2>/dev/null || true
fi
# If no git remote, we'll copy it via SCP later
if [ ! -d "H3VideoGen" ]; then
  mkdir -p H3VideoGen
fi

echo "=== [7/7] Downloading H3 model weights ==="
cd /workspace/ComfyUI/models
mkdir -p unet clip vae

# H3 UNET models (from HuggingFace)
cd unet
if [ ! -f "minimax_h3_fl2va_pruned_int8_convrot.safetensors" ]; then
  echo "Downloading H3 FL2VA UNET..."
  wget -q --show-progress "https://huggingface.co/Kijai/HailuoVideo_comfy/resolve/main/minimax_h3_fl2va_pruned_int8_convrot.safetensors" || true
fi
if [ ! -f "minimax_h3_ref2va_pruned_int8_convrot.safetensors" ]; then
  echo "Downloading H3 R2V UNET..."
  wget -q --show-progress "https://huggingface.co/Kijai/HailuoVideo_comfy/resolve/main/minimax_h3_ref2va_pruned_int8_convrot.safetensors" || true
fi

cd ../clip
if [ ! -f "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" ]; then
  echo "Downloading H3 CLIP..."
  wget -q --show-progress "https://huggingface.co/Kijai/HailuoVideo_comfy/resolve/main/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" || true
fi

cd ../vae
if [ ! -f "minimax_h3_video_vae_fp16.safetensors" ]; then
  echo "Downloading H3 Video VAE..."
  wget -q --show-progress "https://huggingface.co/Kijai/HailuoVideo_comfy/resolve/main/minimax_h3_video_vae_fp16.safetensors" || true
fi
if [ ! -f "minimax_h3_audio_vae_fp32.safetensors" ]; then
  echo "Downloading H3 Audio VAE..."
  wget -q --show-progress "https://huggingface.co/Kijai/HailuoVideo_comfy/resolve/main/minimax_h3_audio_vae_fp32.safetensors" || true
fi

echo "=== SETUP COMPLETE ==="
echo "Models in /workspace/ComfyUI/models:"
ls -la /workspace/ComfyUI/models/unet/ 2>/dev/null
ls -la /workspace/ComfyUI/models/clip/ 2>/dev/null
ls -la /workspace/ComfyUI/models/vae/ 2>/dev/null
