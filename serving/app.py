"""Serve the fine-tuned model as an OpenAI-compatible API on Modal.

Qwen3-8B with Adam's LoRA adapter (data/lora/full/adapter, written by training/train.py),
served by vLLM on an L4 GPU that shuts down when idle. Deploy from the repo root with:

    uv run --group serve modal deploy serving/app.py

Every request needs a Modal proxy token, which Modal checks before a container starts, so
requests without one never wake the GPU. Create a token per client, revocable on its own:

    uv run --group serve modal workspace proxy-tokens create

and send it as "Authorization: Bearer <token id>.<token secret>". Two models are served:
"adam" is the fine-tune, and "qwen3-8b" is the untouched base model, for comparison.

    curl "$URL/v1/chat/completions" \\
      -H "Authorization: Bearer $MODAL_PROXY_TOKEN" -H "Content-Type: application/json" \\
      -d '{"model": "adam", "messages": [{"role": "user", "content": "wyd"}]}'
"""

import json
import subprocess
from pathlib import Path

import modal

BASE_MODEL = "unsloth/Qwen3-8B"
BASE_REVISION = "946bc9ac74a6c1f8cf012497c503a119b2fcf2eb"  # the weights the adapter was trained on
ADAPTER = Path(__file__).resolve().parents[1] / "data" / "lora" / "full" / "adapter"

PORT = 8000
MINUTES = 60  # seconds

image = (
    # CUDA 13.0 matches the driver on Modal's GPU hosts.
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.31.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster weight downloads
            # FlashInfer's sampler compiles its kernels on first use, which added about two
            # minutes to every start from zero. PyTorch's sampler is as fast at this traffic.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .add_local_dir(ADAPTER, "/adapter")
)

# Cache the base weights and vLLM's compiled kernels so only the first start downloads them.
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("adam")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    # Cost limits: never more than one GPU, shut down 5 minutes after the last request.
    max_containers=1,
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,
)
@modal.concurrent(max_inputs=32)  # vLLM batches concurrent requests on the one GPU
@modal.web_server(port=PORT, startup_timeout=10 * MINUTES, requires_proxy_auth=True)
def serve():
    command = [
        "vllm",
        "serve",
        BASE_MODEL,
        "--revision",
        BASE_REVISION,
        "--served-model-name",
        "qwen3-8b",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--enable-lora",
        "--lora-modules",
        "adam=/adapter",
        "--max-lora-rank",
        "16",
        # Training formatted every conversation with thinking off, so serve the same way.
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": False}),
        # Qwen's recommended sampling with thinking off, also used for training's sample
        # replies. Requests can still set their own.
        "--override-generation-config",
        json.dumps({"temperature": 0.7, "top_p": 0.8, "top_k": 20}),
        # Replies are short, so a small context leaves the L4's memory for batching.
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "32",
        # Skip CUDA graph capture: slower generation, but a faster start from zero, which is
        # what most requests to a mostly idle endpoint wait on.
        "--enforce-eager",
    ]

    subprocess.Popen(command)
