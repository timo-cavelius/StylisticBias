# Image Generation Pipeline

A Python project for generating face images with Google Cloud Vertex AI and evaluating multimodal model judgements. It includes:
- **Main pipeline**: Generate face images from structured characteristics in a JSON file
- **Banana pipeline**: Generate variations of base face images with configurable feature modifications
- **Evaluation scripts**: Create statistics, tables, and plots for model comparison and analysis

Generated outputs, local credentials, and model caches are intentionally excluded from version control so the repository can be published safely.

## Features

- 🎨 **Dual Pipeline System**: Main pipeline for direct generation, Banana pipeline for variations
- 📝 **JSON-Based Configuration**: Define characteristics and feature variations in easy-to-edit JSON formats
- 🔄 **Unique Filenames**: Automatically generates unique filenames (UUID + timestamp) to prevent overwriting
- 💾 **Metadata Tracking**: Saves metadata alongside each image including prompts and variations
- 🌐 **Google Cloud Integration**: Uses Vertex AI Gemini models for high-quality image generation
- 🔁 **Automatic Retry Logic**: Up to 5 automatic retry attempts per variation on API failure

## Project Structure

```
StylisticBias/
├── src/
│   ├── main.py                  # Main image generation pipeline
│   ├── banana_pipeline.py       # Variation generation pipeline
│   ├── judgement_pipeline.py    # MLLM judgement runner
│   ├── evaluate_base_faces.py    # Base-face evaluation
│   ├── evaluate_mllm_outputs.py # Full evaluation / statistics pipeline
│   ├── prompt_generator.py      # Prompt generation logic
│   └── image_saver.py           # Image saving and metadata helpers
├── config/
│   ├── characteristics.json     # Base characteristics for main pipeline
│   └── variation_features.json  # Feature variations for banana pipeline
├── base_faces/                  # Local base-face inputs for variation generation
├── output/                      # Generated artifacts (ignored by git)
├── keys/                        # Local service account keys (ignored by git)
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Prerequisites

1. **Google Cloud Account**: You need an active Google Cloud account.
2. **Project with Vertex AI enabled**: Create a project and enable the Vertex AI API.
3. **Service Account**: Create a service account with necessary permissions.
4. **Python 3.8+**: Ensure Python is installed.

## Setup Instructions

### 1. Google Cloud Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Vertex AI API**:
   - Navigate to APIs & Services > Library
   - Search for "Vertex AI API"
   - Click "Enable"
4. Create a service account:
   - Navigate to IAM & Admin > Service Accounts
   - Click "Create Service Account"
   - Grant it the "Vertex AI User" role
   - Create and download a JSON key file
5. Note your **Project ID** from the project dashboard

### 2. Local Environment Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/Image_generation
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and fill in your details:
   ```
   # Google Cloud
   GOOGLE_CLOUD_PROJECT_ID=your-project-id
   GOOGLE_CLOUD_LOCATION=us-central1
   GOOGLE_APPLICATION_CREDENTIALS=keys/vertex-sa.json
   
   # Main Pipeline
   OUTPUT_DIR=output/images
   CHARACTERISTICS_FILE=config/characteristics.json
   MODEL_ID=gemini-2.5-flash-image
   MAX_VARIATIONS=999
   
   # Banana Pipeline
   BANANA_BASE_FACES_DIR=base_faces
   BANANA_OUTPUT_ROOT=output/banana
   BANANA_FEATURES_FILE=config/variation_features.json
   BANANA_MODEL_ID=gemini-2.5-flash-image
   BANANA_MAX_VARIATIONS=999
   TEST_FEATURE=                    # Optional: test single feature only
   ```

5. **Place your service account key**:
   - Save the downloaded JSON key file in the `keys/` directory
   - Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to `keys/vertex-sa.json`

## Usage

### Main Pipeline: Generate Images from Characteristics

Run the main pipeline to generate images from all characteristics in the JSON file:

```bash
python src/main.py
```

### Banana Pipeline: Generate Variations from Base Faces

Generate variations of base face images with different feature modifications:

```bash
python src/banana_pipeline.py
```

**How it works**:
1. Loads base face images from `base_faces/` directory
2. Reads variation features from `config/variation_features.json`
3. For each base face, generates variations by:
   - Creating gender-aware prompts with feature modifications
   - Generating face-focused images for appearance variations
   - Generating full-body images for fashion style variations
4. Saves results under `output/banana/<base_face_name>/`
5. Includes automatic retry logic (up to 5 attempts per variation)

**Test Mode** (single feature):

```bash
export TEST_FEATURE="hair_color"
python src/banana_pipeline.py
```

This generates variations only for the specified feature.

### Evaluation Pipeline: Analyze MLLM Judgements

Run the base-face evaluator (counts + base-only scenario summaries):

```bash
python src/evaluate_base_faces.py --model-folder llava_next
```

Run the full evaluator to compute paired base-vs-variation effects:

```bash
python src/evaluate_mllm_outputs.py --model-folder llava_next
```

Or evaluate all available model folders:

```bash
python src/evaluate_mllm_outputs.py --all-models
```

Outputs are written to `output/evaluation/<model>/` and include:
- `base_faces_counts.png/.csv/.json` (counts for age, gender, ethnicity, body_index)
- `base_faces_probability_scores.csv/.json` (base-only per-face/scenario probabilities across seed/order)
- `base_faces_category_scenario_summary.csv/.json` (per category value and scenario: mean/std for both options)
- `probability_scores.csv/.json` (per face/variation/scenario option probabilities across seeds/orders)
- `paired_deltas.csv/.json` with `delta = score(variation) - score(base)`
- `paired_delta_statistics.json` (mean/std, paired t-test, Wilcoxon, Cohen's d)
- `delta_histogram_overall.png` and `summary.txt`

### Judgement Pipeline: Switch Models (including RunPod vLLM)

You can now switch judgement backends with one flag:

```bash
python src/judgement_pipeline.py --model vllm
python src/judgement_pipeline.py --model pixtral
python src/judgement_pipeline.py --model phi-4
python src/judgement_pipeline.py --model gemma4
python src/judgement_pipeline.py --model gemma3
```

Supported values for `--model` (or `JUDGE_MODEL_TYPE`):
- `vllm` (OpenAI-compatible vLLM endpoint, e.g. RunPod)
- `pixtral` (OpenAI-compatible vLLM endpoint with Pixtral defaults)
- `gemma4` (Vertex AI endpoint, output subdir defaults to `gemma4`)
- `gemma3` (Vertex AI endpoint)
- `phi-4` (Azure Foundry Phi-4)
- `phi4` (local Phi-4 multimodal)
- `llava`
- `ollama`
- `dummy`

Useful optional flags:

```bash
python src/judgement_pipeline.py \
   --model vllm \
   --output-subdir gemma3_12b_runpod \
   --max-workers 8
```

RunPod vLLM environment variables:

```bash
export JUDGE_MODEL_TYPE=vllm
export VLLM_BASE_URL="https://<your-runpod-endpoint>"
export VLLM_MODEL_ID="google/gemma-3-12b-it"
export VLLM_API_KEY="<optional-if-endpoint-requires-it>"
export VLLM_CHAT_COMPLETIONS_PATH="/v1/chat/completions"  # optional override
export VLLM_TEMPERATURE=0.2
export VLLM_MAX_TOKENS=16
export VLLM_TIMEOUT_SECONDS=120
export VLLM_MAX_RETRIES=5
```

RunPod Pixtral environment variables (recommended for your pod):

```bash
export JUDGE_MODEL_TYPE=pixtral
export PIXTRAL_BASE_URL="https://j1xx0ykqxf0gqw-8000.proxy.runpod.net"
export PIXTRAL_MODEL_ID="mistralai/Pixtral-12B-2409"
export PIXTRAL_API_KEY="<optional-if-endpoint-requires-it>"
export PIXTRAL_CHAT_COMPLETIONS_PATH="/v1/chat/completions/render"  # Trelis/RunPod images endpoint
export PIXTRAL_TEMPERATURE=0.2
export PIXTRAL_MAX_TOKENS=16
export PIXTRAL_TIMEOUT_SECONDS=120
export PIXTRAL_MAX_RETRIES=5
```

Then run:

```bash
python src/judgement_pipeline.py --model pixtral --output-subdir pixtral
```

RunPod/Trelis container checklist:
- Set Docker command model flags for Pixtral.
- Keep port `8000` exposed (or tunnel via SSH if you prefer).
- Add `HUGGING_FACE_HUB_TOKEN` in pod environment variables when model access requires Hugging Face auth.
- Use endpoint format: `https://<pod-id>-8000.proxy.runpod.net`.

Recommended vLLM container launch flags for Pixtral on RunPod:

```bash
--model mistralai/Pixtral-12B-2409 \
--tokenizer-mode mistral \
--limit-mm-per-prompt image=4 \
--max-model-len 16384 \
--port 8000
```

OpenAI-compatible request test (RunPod proxy):

```bash
curl --location 'https://j1xx0ykqxf0gqw-8000.proxy.runpod.net/v1/chat/completions' \
   --header 'Content-Type: application/json' \
   --header 'Authorization: Bearer <token-if-required>' \
   --data '{
      "model": "mistralai/Pixtral-12B-2409",
      "messages": [
         {
            "role": "user",
            "content": [
               {"type": "text", "text": "Describe this image in detail please."},
               {"type": "image_url", "image_url": {"url": "https://s3.amazonaws.com/cms.ipressroom.com/338/files/201808/5b894ee1a138352221103195_A680%7Ejogging-edit/A680%7Ejogging-edit_hero.jpg"}}
            ]
         }
      ]
   }'
```

Notes:
- `VLLM_BASE_URL` can be either the base URL or a `/v1` URL.
- If your server uses a non-standard route, set `VLLM_CHAT_COMPLETIONS_PATH` explicitly.
- Outputs are written to `output/judgements/<output-subdir>/`.
- If `--output-subdir` is not set, the default for vLLM is `vllm`.
- If `--model pixtral` is used and `--output-subdir` is not set, the default is `pixtral`.
- For this project pipeline, local face images are sent as base64 data URLs by the client, so you do not need public image URLs during batch judgement runs.
- Some RunPod/Trelis images expose only `/v1/chat/completions/render`; the client now auto-tries both standard and `/render` routes.
- Pixtral auth uses only `PIXTRAL_API_KEY` (it no longer falls back to `VLLM_API_KEY`) to avoid cross-pod credential mismatch.

### Customize Main Pipeline Characteristics

Edit [config/characteristics.json](config/characteristics.json) to define your own face characteristics:

```json
[
  {
    "age": "young adult",
    "gender": "female",
    "ethnicity": "Asian",
    "hair_color": "black",
    "hair_style": "long straight",
    "eye_color": "brown",
    "skin_tone": "medium",
    "facial_features": "oval face",
    "expression": "smiling"
  }
]
```

### Customize Banana Pipeline Variations

Edit [config/variation_features.json](config/variation_features.json) to define feature variations:

```json
{
  "skin_irregularities": ["clear skin", "light acne", "freckles"],
  "hair_color": ["black", "brown", "blonde", "red"],
  "hair_length": ["short", "medium", "long"],
  "hair_style": ["straight", "wavy", "curly"],
  "eyewear": ["no glasses", "glasses", "sunglasses"],
  "makeup_female": ["no makeup", "natural", "bold"],
  "facial_hair_male": ["clean shaven", "stubble", "beard"],
  "fashion_style": ["casual", "formal", "athletic"],
  "accessories": ["none", "necklace", "scarf"]
}
```

**Retry Behavior**:
- Each variation attempt automatically retries up to 5 times on failure
- Waits between attempts (exponential backoff: 1s, 2s, 4s, 8s, 16s max)
- After 5 failed attempts, skips to the next variation
- Logs all failed attempts for debugging

### Output Structure

**Main Pipeline** saves to `output/images/` by default:
- **Images**: `face_1_20251222_143022_a3f9b2c1.png`
- **Metadata**: `face_1_20251222_143022_a3f9b2c1_metadata.json`

Metadata includes:
- Original characteristics
- Generated prompt
- Timestamp
- Image path

**Banana Pipeline** saves to `output/banana/<base_face_name>/`:
- Original base image
- Original metadata
- Generated variations:
  - Face-focused: `<name>_face_<idx>.png`
  - Full-body: `<name>_body_<idx>.png`
- Metadata for each variation with prompt and feature changes

## Troubleshooting

### Authentication Errors

If you get authentication errors:
1. Verify `GOOGLE_APPLICATION_CREDENTIALS` path is correct
2. Ensure the service account has "Vertex AI User" role
3. Check that the Vertex AI API is enabled in your project

### API Quota Errors

Google Cloud has usage limits. If you hit quota limits:
1. Check your quota in the Google Cloud Console
2. Request a quota increase if needed
3. Reduce the number of images generated per run

### Module Import Errors

If you get import errors:
```bash
# Run from the repository root
python src/main.py
```

### Rate Limiting / API Quota

The pipeline includes automatic retry logic:
- Detects 429 (rate limit) and RESOURCE_EXHAUSTED errors
- Retries up to 5 times per variation with exponential backoff
- Logs each attempt for monitoring

## Cost Considerations

Google Cloud Vertex AI charges per image generation. Check the [Vertex AI pricing page](https://cloud.google.com/vertex-ai/pricing) for current rates. Monitor your usage in the Google Cloud Console.

## Development

### Adding New Features

- **Main pipeline prompts**: Edit [src/prompt_generator.py](src/prompt_generator.py)
- **Banana pipeline prompts**: Edit functions in [src/banana_pipeline.py](src/banana_pipeline.py) like `build_face_prompt()` and `build_body_prompt()`
- **Image saving**: Modify [src/image_saver.py](src/image_saver.py)
- **Retry logic**: Adjust `call_model()` function parameters in [src/banana_pipeline.py](src/banana_pipeline.py)

### Testing

**Main pipeline - single test image**:
```python
from prompt_generator import generate_prompt

prompt = generate_prompt({"age": "young", "gender": "female", "expression": "smiling"})
print(prompt)
```

**Banana pipeline - test single feature**:
```bash
export TEST_FEATURE="hair_color"
python src/banana_pipeline.py
```

## Repository Hygiene

Before publishing the repository publicly, keep the following out of version control:

- Local secrets and credentials in `keys/` and `.env`
- Generated image and evaluation outputs under `output/`
- Local model assets in `src/LMML_models/`
- Interpreter caches such as `__pycache__/` and `.venv/`

The repository's `.gitignore` is configured for these generated artifacts and local-only files.

## License

This project is for educational and research purposes.

## Support

For issues related to:
- **Google Cloud**: Check [Google Cloud Documentation](https://cloud.google.com/docs)
- **Vertex AI**: See [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- **This pipeline**: Open an issue in the repository
