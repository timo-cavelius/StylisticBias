from pathlib import Path


class JudgeModel:
    def generate(self, image_path: Path, prompt: str, seed: int) -> str:
        raise NotImplementedError
