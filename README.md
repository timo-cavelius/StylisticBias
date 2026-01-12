# Image Generation Pipeline

A Python-based pipeline system for generating face images using Google Cloud Vertex AI. Includes two pipelines:
- **Main Pipeline**: Generate face images from structured characteristics in a JSON file
- **Banana Pipeline**: Generate variations of base face images with configurable feature modifications

## Features

- 🎨 **Dual Pipeline System**: Main pipeline for direct generation, Banana pipeline for variations
- 📝 **JSON-Based Configuration**: Define characteristics and feature variations in easy-to-edit JSON formats
- 🔄 **Unique Filenames**: Automatically generates unique filenames (UUID + timestamp) to prevent overwriting
- 💾 **Metadata Tracking**: Saves metadata alongside each image including prompts and variations
- 🌐 **Google Cloud Integration**: Uses Vertex AI Gemini models for high-quality image generation
- 🔁 **Automatic Retry Logic**: Up to 5 automatic retry attempts per variation on API failure

## Project Structure

```
Image_generation/
├── src/
│   ├── main.py                  # Main pipeline script
│   ├── banana_pipeline.py       # Banana pipeline (variations generator)
│   ├── prompt_generator.py      # Prompt generation logic
│   ├── image_saver.py           # Image saving with unique names
│   └── __pycache__/             # Python cache
├── base_faces/                  # Base face images for banana pipeline
├── output/
│   ├── images/                  # Main pipeline output (metadata)
│   └── banana/                  # Banana pipeline output by base face
├── config/
│   ├── characteristics.json     # Base characteristics for main pipeline
│   └── variation_features.json  # Feature variations for banana pipeline
├── keys/
│   └── vertex-sa.json          # Google Cloud service account credentials
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Prerequisites

1. **Google Cloud Account**: You need an active Google Cloud account
2. **Project with Vertex AI enabled**: Create a project and enable the Vertex AI API
3. **Service Account**: Create a service account with necessary permissions
4. **Python 3.8+**: Ensure Python is installed

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
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
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
cd src
python main.py
```

### Banana Pipeline: Generate Variations from Base Faces

Generate variations of base face images with different feature modifications:

```bash
cd src
python banana_pipeline.py
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
python banana_pipeline.py
```

This generates variations only for the specified feature.

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

**Main Pipeline** saves to `output/images/`:
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
# Make sure you're in the src directory
cd src
python main.py
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
cd src
python banana_pipeline.py
```

## License

This project is for educational and research purposes.

## Support

For issues related to:
- **Google Cloud**: Check [Google Cloud Documentation](https://cloud.google.com/docs)
- **Vertex AI**: See [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- **This pipeline**: Open an issue in the repository
