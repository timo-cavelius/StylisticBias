from contextlib import contextmanager
from pathlib import Path
import os

from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor
from transformers import cache_utils

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover
    from transformers import AutoModelForCausalLM as AutoModelForVision2Seq

from .base import JudgeModel


class Phi4MultimodalJudge(JudgeModel):
    def __init__(
        self,
        model_id: str = "microsoft/Phi-4-multimodal-instruct",
        max_new_tokens: int = 32,
        temperature: float = 0.2,
    ) -> None:
        os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "eager")

        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        except Exception:
            pass
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if not hasattr(cache_utils, "SlidingWindowCache"):
            class SlidingWindowCache(cache_utils.StaticCache):
                pass

            cache_utils.SlidingWindowCache = SlidingWindowCache

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

        with default_device("cpu"):
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                config=config,
                **load_kwargs,
            )

        if torch.backends.mps.is_available():
            self.device = "mps"
            self.model = self.model.to("mps")
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.model = self.model.to("cuda")
        else:
            self.device = "cpu"

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )

    def generate(self, image_path: Path, prompt: str, seed: int) -> str:
        torch.manual_seed(seed)

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]

        if hasattr(self.processor, "apply_chat_template"):
            chat_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=chat_prompt,
                images=image,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
        )
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return output_text
