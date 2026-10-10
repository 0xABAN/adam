"""Serve the fine-tuned model as an OpenAI-compatible API on Modal.

After training (training/train.py writes data/lora/full/adapter), merge the adapter into
Qwen3-8B, store it in FP8 on a Modal Volume, and deploy. From the repo root:

    uv run --group serve modal run serving/app.py::merge
    uv run --group serve modal deploy serving/app.py

vLLM serves the model as "adam" on an L4 GPU that shuts down when idle. Every request needs
a Modal proxy token, which Modal checks before a container starts, so requests without one
never wake the GPU. Create a token per client, revocable on its own:

    uv run --group serve modal workspace proxy-tokens create

and send it as "Authorization: Bearer <token id>.<token secret>":

    curl "$MODAL_URL/v1/chat/completions" \\
      -H "Authorization: Bearer $MODAL_PROXY_TOKEN" -H "Content-Type: application/json" \\
      -d '{"model": "adam", "messages": [{"role": "user", "content": "wyd"}]}'

Measured on an L4: about 90 s to answer from a cold start, about 0.6 s for a typical reply
once warm. GPU memory snapshots were tried and dropped: restores rarely happened, and
creating them failed some starts.
"""

import json
import subprocess
from pathlib import Path

import modal

BASE_MODEL = "unsloth/Qwen3-8B"
BASE_REVISION = "946bc9ac74a6c1f8cf012497c503a119b2fcf2eb"  # the weights the adapter was trained on
ADAPTER = Path(__file__).resolve().parents[1] / "data" / "lora" / "full" / "adapter"
MODEL_DIR = "/models/adam"  # the merged FP8 model, on the models Volume

PORT = 8000
MINUTES = 60  # seconds

models = modal.Volume.from_name("adam-models", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("adam")

merge_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("transformers==5.17.0", "peft==0.21.2", "llmcompressor==0.14.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster base weight downloads
    .add_local_dir(ADAPTER, "/adapter")
)


@app.function(
    image=merge_image,
    cpu=8,
    memory=64 * 1024,
    volumes={"/models": models, "/root/.cache/huggingface": hf_cache},
    timeout=30 * MINUTES,
)
def merge():
    """Merge the LoRA adapter into the base model and save it in FP8 for serving.

    FP8 halves the bytes the GPU reads per generated token, so replies come out about 40%
    faster than in bf16, for a holdout loss within 0.005 of bf16's. Serving merged weights
    instead of base + adapter also avoids LoRA's per-token overhead.
    """
    import torch
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, "/adapter").merge_and_unload()

    # FP8 weights with FP8 activations scaled per token at run time, so no calibration data is
    # needed. The output layer stays in bf16, as llm-compressor recommends.
    oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]))

    model.save_pretrained(MODEL_DIR)
    AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION).save_pretrained(MODEL_DIR)
    models.commit()


vllm_image = (
    # CUDA 13.0 matches the driver on Modal's GPU hosts.
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.31.0")
    # FlashInfer's sampler compiles its kernels on first use, which added about two minutes
    # to every start from zero. PyTorch's sampler is as fast at this traffic.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)


@app.function(
    image=vllm_image,
    gpu="L4",
    # vllm_cache holds the compiled CUDA graphs. Compiling takes about 100 s, and only
    # happens on the first start after a vLLM or configuration change.
    volumes={"/models": models, "/root/.cache/vllm": vllm_cache},
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
        MODEL_DIR,
        "--served-model-name",
        "adam",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
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
    ]

    subprocess.Popen(command)
