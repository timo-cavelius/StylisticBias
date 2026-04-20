import base64
import mimetypes
import os
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import JudgeModel


class VllmOpenAIJudge(JudgeModel):
    def __init__(
        self,
        base_url: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        chat_completions_path: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 16,
        timeout_seconds: int = 120,
        max_retries: int = 5,
        backoff_base_seconds: float = 1.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL") or "").strip().rstrip("/")
        self.model_id = (model_id or os.getenv("VLLM_MODEL_ID") or "google/gemma-3-12b-it").strip()
        self.api_key = (api_key or os.getenv("VLLM_API_KEY") or "").strip()
        self.chat_completions_path = (
            chat_completions_path or os.getenv("VLLM_CHAT_COMPLETIONS_PATH") or ""
        ).strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds

        if not self.base_url:
            raise ValueError("VLLM_BASE_URL is required")

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

    def _join_url(self, path: str) -> str:
        normalized_path = path.strip()
        if not normalized_path.startswith("/"):
            normalized_path = "/" + normalized_path
        return f"{self.base_url}{normalized_path}"

    def _candidate_urls(self) -> list[str]:
        if self.base_url.endswith("/chat/completions"):
            candidates = [self.base_url, f"{self.base_url}/render"]
            unique_candidates = []
            seen = set()
            for url in candidates:
                if url not in seen:
                    unique_candidates.append(url)
                    seen.add(url)
            return unique_candidates

        if self.base_url.endswith("/chat/completions/render"):
            return [self.base_url]

        if self.chat_completions_path:
            explicit = self._join_url(self.chat_completions_path)
            candidates = [explicit]
            if explicit.endswith("/chat/completions"):
                candidates.append(f"{explicit}/render")
            elif explicit.endswith("/chat/completions/render"):
                candidates.append(explicit[: -len("/render")])

            # Add OpenAI-prefixed alternatives for proxies that mount under /openai.
            if "/openai/" not in explicit:
                if "/v1/" in explicit:
                    candidates.append(explicit.replace("/v1/", "/openai/v1/", 1))
                else:
                    candidates.append(f"{self.base_url}/openai{self.chat_completions_path if self.chat_completions_path.startswith('/') else '/' + self.chat_completions_path}")
            else:
                candidates.append(explicit.replace("/openai/", "/", 1))

            # Include optional trailing slash variants.
            for url in list(candidates):
                if not url.endswith("/"):
                    candidates.append(url + "/")

            unique_candidates = []
            seen = set()
            for url in candidates:
                if url not in seen:
                    unique_candidates.append(url)
                    seen.add(url)
            return unique_candidates

        candidates = []
        if self.base_url.endswith("/v1"):
            candidates.append(f"{self.base_url}/chat/completions")
            candidates.append(f"{self.base_url}/chat/completions/render")
        else:
            candidates.append(f"{self.base_url}/v1/chat/completions")
            candidates.append(f"{self.base_url}/v1/chat/completions/render")
            candidates.append(f"{self.base_url}/openai/v1/chat/completions")
            candidates.append(f"{self.base_url}/openai/v1/chat/completions/render")
            candidates.append(f"{self.base_url}/chat/completions")
            candidates.append(f"{self.base_url}/chat/completions/render")

        unique_candidates = []
        seen = set()
        for url in candidates:
            if url not in seen:
                unique_candidates.append(url)
                seen.add(url)
        return unique_candidates

    def _image_to_data_url(self, image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            mime_type = "image/png"

        with image_path.open("rb") as handle:
            image_bytes = handle.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{image_b64}"

    def generate(self, image_path: Path, prompt: str, seed: int) -> str:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_to_data_url(image_path)
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

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        candidate_urls = self._candidate_urls()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            saw_404 = False
            for url in candidate_urls:
                try:
                    response = self.session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )

                    if response.status_code == 404:
                        saw_404 = True
                        continue

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
                    content = message.get("content", "")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(str(item.get("text", "")))
                        return "\n".join(part for part in text_parts if part)
                    return str(content)
                except Exception as exc:
                    last_error = exc

            if saw_404:
                raise RuntimeError(
                    "vLLM endpoint returned 404 for all known chat routes. "
                    "Set VLLM_CHAT_COMPLETIONS_PATH to your server route "
                    "(e.g. /v1/chat/completions or /v1/chat/completions/render). "
                    f"Tried: {', '.join(candidate_urls)}"
                )

            if attempt >= self.max_retries:
                break
            sleep_s = min(self.backoff_base_seconds * (2 ** (attempt - 1)), 30)
            time.sleep(sleep_s)

        raise RuntimeError(f"vLLM request failed after {self.max_retries} attempts across routes {candidate_urls}: {last_error}")
