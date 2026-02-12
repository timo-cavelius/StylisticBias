import os
from contextlib import contextmanager

os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "eager")

import torch
from dotenv import load_dotenv
from transformers import AutoConfig, AutoProcessor, cache_utils

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover
    from transformers import AutoModelForCausalLM as AutoModelForVision2Seq

try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
except Exception:
    pass

model_id = "microsoft/Phi-4-multimodal-instruct"

load_dotenv()

if not hasattr(cache_utils, "SlidingWindowCache"):
    class SlidingWindowCache(cache_utils.StaticCache):
        pass

    cache_utils.SlidingWindowCache = SlidingWindowCache

if os.getenv("HF_TOKEN") is None:
    print("Warning: HF_TOKEN not set. Downloads may be slow or rate-limited.")

processor = AutoProcessor.from_pretrained(
    model_id, trust_remote_code=True, use_fast=False
)

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config._attn_implementation = "eager"
if hasattr(config, "attn_implementation"):
    config.attn_implementation = "eager"
if hasattr(config, "rope_scaling") and config.rope_scaling:
    orig = config.rope_scaling.get("original_max_position_embeddings")
    if orig:
        config.rope_scaling["factor"] = config.max_position_embeddings / orig

load_kwargs = {
    "torch_dtype": torch.float16 if torch.backends.mps.is_available() else torch.float32,
    "trust_remote_code": True,
    "low_cpu_mem_usage": False,
    "device_map": None,
    "attn_implementation": "eager",
}


@contextmanager
def default_device(device: str):
    if hasattr(torch, "get_default_device") and hasattr(torch, "set_default_device"):
        previous = torch.get_default_device()
        torch.set_default_device(device)
        try:
            yield
        finally:
            torch.set_default_device(previous)
    else:
        yield

print("Loading model into system RAM (Forcing CPU default device)...")
with default_device("cpu"):
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        config=config,
        **load_kwargs,
    )

if torch.backends.mps.is_available():
    print("Moving model to Apple Silicon GPU (MPS)...")
    model = model.to("mps")

print("Downloaded:", model_id)
