from pathlib import Path

from .base import JudgeModel


class DummyJudge(JudgeModel):
    def generate(self, image_path: Path, prompt: str, seed: int) -> str:
        return "a"
