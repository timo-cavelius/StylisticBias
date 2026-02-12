import base64
import os
from pathlib import Path
from typing import Any, Dict

import requests
from requests.exceptions import ReadTimeout

from .base import JudgeModel


class OllamaLlavaJudge(JudgeModel):
    def __init__(
        self,
        model_id: str = "llava:7b",
        host: str | None = None,
        timeout: int = 300,
        temperature: float = 0.2,
    ) -> None:
        self.model_id = model_id
        if host is None:
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def generate(self, image_path: Path, prompt: str, seed: int) -> str:
        with image_path.open("rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "keep_alive": "0s",
            "options": {
                "temperature": self.temperature,
                "seed": seed,
            },
        }

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
        except ReadTimeout:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout * 2,
            )
        if response.status_code == 404:
            chat_payload: Dict[str, Any] = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "images": [image_b64],
                "keep_alive": "0s",
                "options": {
                    "temperature": self.temperature,
                    "seed": seed,
                },
            }
            response = requests.post(
                f"{self.host}/api/chat",
                json=chat_payload,
                timeout=self.timeout,
            )
        if response.status_code == 404:
            openai_payload: Dict[str, Any] = {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                "stream": False,
                "keep_alive": "0s",
                "temperature": self.temperature,
            }
            response = requests.post(
                f"{self.host}/v1/chat/completions",
                json=openai_payload,
                timeout=self.timeout,
            )
        response.raise_for_status()
        data = response.json()
        if "response" in data:
            return data.get("response", "")
        if "choices" in data:
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            return message.get("content", "")
        message = data.get("message", {})
        return message.get("content", "")
