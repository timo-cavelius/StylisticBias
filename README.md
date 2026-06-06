# StylisticBias

![StylisticBias pipeline overview](assets/final_pipeline.png)

StylisticBias is a controlled benchmark for measuring attribute-level social bias in multimodal large language models. It generates base faces, creates single-attribute variations, runs scenario-based judgements, and evaluates how visual cues shift model answers while keeping identity fixed.

## Abstract

Multimodal large language models (MLLMs) are increasingly deployed in personally and societally consequential settings, yet the visual cues that shape how these models judge people remain poorly understood. Prior work often compares different (groups of) individuals, making it difficult to separate appearance effects from identity differences. We introduce StylisticBias, a controlled benchmark for evaluating attribute-level social bias in MLLMs. We generate 500 photorealistic base faces and create about 50 single-attribute variations per face, producing about 25K images. This design keeps identity fixed and changes one visual attribute at a time. It lets us measure how specific cues shift model judgments. We evaluate six MLLMs across 25 binary social judgment scenarios. We find that age and body type dominate identity-level effects, while fashion style and other visual cues drive the largest attribute-level shifts. We further find that about 15 attributes account for nearly 80% of the total variation, showing that bias is concentrated in a small set of visual cues. Sensitivity is strongest in judgments that are semantically aligned with appearance, especially socioeconomic and style-related judgments. We release StylisticBias as a benchmark for fine-grained bias evaluation in multimodal models.

## Pipeline

1. **Generation** uses `src/generation/main.py` to create base faces from `config/characteristics.json`.
2. **Variation building** uses `src/generation/banana_pipeline.py` to generate single-attribute edits from `base_faces/` and `config/variation_features.json`.
3. **Judgement** uses `src/judgement/judgement_pipeline.py` and runs each scenario with 4 random orders and 3 seeds for stability.
4. **Evaluation** summarizes paired base-vs-variation effects in `output/evaluation/`.

Generation models: the main pipeline uses Google Vertex AI Imagen 4; the banana/variation pipeline uses Nano Banana (Gemini 2.5 Flash Image).

Judgement extension: the judgement pipeline can be configured with different scenario lists — see `config/judgement_scenarios_short.json`, `_medium.json`, and `_long.json` to shorten, lengthen, or extend runs.

## What It Produces

- `output/images/`: base-face images and metadata
- `output/banana/`: face and full-body variation sets
- `output/judgements/`: raw MLLM responses for each scenario
- `output/evaluation/`: plots, tables, and statistics

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Configure Google Cloud credentials in `.env`, then run the pipeline you need from the repository root.

## Common Commands

```bash
python src/generation/main.py
python src/generation/banana_pipeline.py
python src/judgement/judgement_pipeline.py --model vllm
python src/evaluation/evaluate_mllm_outputs.py --all-models
```

## Dataset

The final dataset has been uploaded to Hugging Face (placeholder): [stylistic-bias/stylistic-bias-dataset](https://huggingface.co/datasets/stylistic-bias/stylistic-bias-dataset).

## Notes

- Local outputs, caches, and credentials are ignored by git.

## BibTeX
[](https://github.com/Picsart-AI-Research/OpenBias#bibtex)
Please cite our work if you find it useful:

```bibtex
@misc{stylisticbias2026,
	title        = {StylisticBias: A Controlled Benchmark for Attribute-Level Social Bias in Multimodal Large Language Models},
	author       = {Timo Cavelius},
	year         = {2026},
	note         = {Repository and benchmark release}
}
```
