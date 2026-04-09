# Installing CUDA 13.2 and Flash Attention 3

## CUDA 13.2 runtime libraries

The compute nodes have CUDA 12.8 installed. To use CUDA 13.2, download the
toolkit and install to a shared directory — no root required.

1. Download `cuda_13.2.0_595.45.04_linux.run` from NVIDIA.

2. Check that the GPU driver supports CUDA 13.x:
   ```
   srun --partition=hpc-mid --gres=gpu:1 nvidia-smi
   ```
   The top-right should show "CUDA Version: 13.x".

3. Install the toolkit (no driver, no GUI):
   ```
   sh cuda_13.2.0_595.45.04_linux.run \
       --toolkit \
       --toolkitpath=/mnt/vast/model_cache_kb/cuda-13.2 \
       --silent \
       --no-drm \
       --no-man-page \
       --no-opengl-libs
   ```

4. Add to `.env`:
   ```
   export LD_LIBRARY_PATH=/mnt/vast/model_cache_kb/cuda-13.2/lib64:$LD_LIBRARY_PATH
   export PATH=/mnt/vast/model_cache_kb/cuda-13.2/bin:$PATH
   ```

## Flash Attention 3

FA3 must be built on a GPU node (needs CUDA to compile kernels).

`.build_fa.env` disables kernels we don't need for Qwen on Hopper (BF16, hdim128, SM90 only), which cuts build time significantly.

1. Get a GPU node:
   ```
   srun --partition=hpc-mid --gres=gpu:1 --time=00:30:00 --pty bash
   ```

2. Build:
   ```
   cd /mnt/home/kimbo/prime-rl
   source .build_fa.env
   source .venv/bin/activate
   uv pip install flash_attn_3 --no-build-isolation
   ```

3. Verify:
   ```
   python -c "from transformers.utils.import_utils import is_flash_attn_3_available; print(is_flash_attn_3_available())"
   ```
   Should print `True`.

4. Use in config:
   ```toml
   [trainer.model]
   attn = "flash_attention_3"
   ```
