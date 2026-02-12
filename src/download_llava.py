import os

import torch
from dotenv import load_dotenv
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

load_dotenv()

if os.getenv("HF_TOKEN") is None:
    print("Warning: HF_TOKEN not set. Downloads may be slow or rate-limited.")

processor = LlavaNextProcessor.from_pretrained(model_id)

load_kwargs = {
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    "device_map": "auto",
}

model = LlavaNextForConditionalGeneration.from_pretrained(model_id, **load_kwargs)

print("Downloaded:", model_id)
