import base64
import os
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import JudgeModel


class AzureFoundryPhi4Judge(JudgeModel):
    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        api_version: str = "2024-10-21",
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: int = 120,
        max_retries: int = 5,
        backoff_base_seconds: float = 1.0,
    ) -> None:
        self.endpoint = endpoint or os.getenv("AZURE_FOUNDRY_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_FOUNDRY_API_KEY")
        self.deployment = deployment or os.getenv("AZURE_FOUNDRY_DEPLOYMENT")
        self.api_version = os.getenv("AZURE_FOUNDRY_API_VERSION", api_version)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds

        if not self.endpoint:
            raise ValueError("AZURE_FOUNDRY_ENDPOINT is required")
        if not self.api_key:
            raise ValueError("AZURE_FOUNDRY_API_KEY is required")
        if not self.deployment:
            raise ValueError("AZURE_FOUNDRY_DEPLOYMENT is required")

        self.endpoint = self.endpoint.rstrip("/")
        self.model_id = f"azure-phi4-{self.deployment}"

        self.session = requests.Session()
        retry = Retry(
            total=0,
            connect=0,
            read=0,
            status=0,
            backoff_factor=0,
            allowed_methods=frozenset(["POST"]),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _build_url(self) -> str:
        return (
            f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )

    def generate(self, image_path: Path, prompt: str, seed: int) -> str:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": seed,
        }

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    self._build_url(),
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )

                if response.status_code in {408, 429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"Retryable status {response.status_code}: {response.text}",
                        response=response,
                    )

                response.raise_for_status()

                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    return ""
                message = choices[0].get("message") or {}
                return str(message.get("content", ""))
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_s = min(self.backoff_base_seconds * (2 ** (attempt - 1)), 30)
                time.sleep(sleep_s)

        raise RuntimeError(f"Azure Foundry request failed after {self.max_retries} attempts: {last_error}")
