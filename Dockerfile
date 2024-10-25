# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --upgrade -r requirements.txt

# Install runpod
RUN pip3 install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Stage 2: Download models
FROM base as downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE

# Change working directory to ComfyUI
WORKDIR /comfyui

# Download checkpoints/vae/LoRA to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    elif [ "$MODEL_TYPE" = "sd3" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    #edit: add models
    elif [ "$MODEL_TYPE" = "relight" ]; then \
      wget -O models/checkpoints/CyberRealisticXLPlay_V1.0-VAE.safetensors https://huggingface.co/cyberdelia/CyberRealsticXL/resolve/main/CyberRealisticXLPlay_V1.0-VAE.safetensors && \
      wget -O models/checkpoints/CyberRealistic_V5_FP32.safetensors https://huggingface.co/cyberdelia/CyberRealistic/resolve/main/CyberRealistic_V5_FP32.safetensors && \
      wget -O models/unet/iclight_sd15_fbc.safetensors https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors; \
    fi

# edit: add custom nodes
RUN git clone https://github.com/Fannovel16/ComfyUI-Video-Matting.git custom_nodes/ComfyUI-Video-Matting
RUN pip3 install -r custom_nodes/ComfyUI-Video-Matting/requirements.txt
RUN git clone https://github.com/cubiq/ComfyUI_essentials.git custom_nodes/ComfyUI_essentials
RUN pip3 install -r custom_nodes/ComfyUI_essentials/requirements.txt
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git custom_nodes/ComfyUI-KJNodes
RUN pip3 install -r custom_nodes/ComfyUI-KJNodes/requirements.txt
RUN git clone https://github.com/jamesWalker55/comfyui-various.git custom_nodes/comfyui-various
RUN pip3 install -r custom_nodes/comfyui-various/requirements.txt
RUN git clone https://github.com/chflame163/ComfyUI_LayerStyle.git custom_nodes/ComfyUI_LayerStyle
RUN pip3 install -r custom_nodes/ComfyUI_LayerStyle/requirements.txt
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git custom_nodes/ComfyUI-Impact-Pack
RUN pip3 install -r custom_nodes/ComfyUI-Impact-Pack/requirements.txt
RUN git clone https://github.com/rgthree/rgthree-comfy.git custom_nodes/rgthree-comfy
RUN pip3 install -r custom_nodes/rgthree-comfy/requirements.txt
RUN git clone https://github.com/kijai/ComfyUI-IC-Light.git custom_nodes/ComfyUI-IC-Light
RUN pip3 install -r custom_nodes/ComfyUI-IC-Light/requirements.txt
RUN git clone https://github.com/huagetai/ComfyUI-Gaffer.git custom_nodes/ComfyUI-Gaffer
RUN pip3 install -r custom_nodes/ComfyUI-Gaffer/requirements.txt
RUN git clone https://github.com/Youngryeol/ImageUtil.git custom_nodes/ImageUtil
#RUN pip3 install -r custom_nodes/ImageUtil/requirements.txt


# Stage 3: Final image
FROM base as final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start the container
CMD /start.sh