from pathlib import Path

from PIL import Image
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

from .base import JudgeModel


class LLaVANextJudge(JudgeModel):
    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        max_new_tokens: int = 16,
        temperature: float = 0.2,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="auto"
            )
        else:
            self.device = "cpu"
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float32
            ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

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
        chat_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=chat_prompt,
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
