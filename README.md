# Image Generation Pipeline

A Python-based pipeline for generating face images using Google Cloud Vertex AI. This tool reads facial characteristics from a JSON file, generates descriptive prompts, and creates photorealistic face images using Google's image generation models.

## Features

- 🎨 **Automated Image Generation**: Generate face images from structured characteristics
- 📝 **JSON-Based Configuration**: Define characteristics in an easy-to-edit JSON format
- 🔄 **Unique Filenames**: Automatically generates unique filenames (UUID + timestamp) to prevent overwriting
- 💾 **Metadata Tracking**: Saves metadata alongside each image for reproducibility
- 🌐 **Google Cloud Integration**: Uses Vertex AI for high-quality image generation

## Project Structure

```
Image_generation/
├── src/
│   ├── main.py              # Main pipeline script
│   ├── prompt_generator.py  # Prompt generation logic
│   └── image_saver.py       # Image saving with unique names
├── output/
│   └── images/              # Generated images folder (created automatically)
├── config/
│   └── characteristics.json # Base characteristics data
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
└── README.md                # This file
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
   GOOGLE_CLOUD_PROJECT_ID=your-project-id
   GOOGLE_CLOUD_LOCATION=us-central1
   GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
   OUTPUT_DIR=output/images
   CHARACTERISTICS_FILE=config/characteristics.json
   ```

5. **Place your service account key**:
   - Save the downloaded JSON key file in the project directory
   - Update `GOOGLE_APPLICATION_CREDENTIALS` in `.env` with the path

## Usage

### Basic Usage

Run the pipeline to generate images from all characteristics in the JSON file:

```bash
cd src
python main.py
```

### Customize Characteristics

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

**Available fields**:
- `age`: e.g., "young", "middle-aged", "senior"
- `gender`: e.g., "male", "female", "person"
- `ethnicity`: e.g., "Asian", "Caucasian", "African", "Hispanic"
- `hair_color`: e.g., "black", "brown", "blonde", "gray"
- `hair_style`: e.g., "long", "short", "curly", "straight"
- `eye_color`: e.g., "brown", "blue", "green", "hazel"
- `skin_tone`: e.g., "fair", "medium", "olive", "dark"
- `facial_features`: e.g., "oval face", "square jaw", "round face"
- `expression`: e.g., "smiling", "neutral", "confident"

### Output

Generated images will be saved in the `output/images/` directory with:
- **Unique filenames**: `face_1_20251222_143022_a3f9b2c1.png`
- **Metadata files**: `face_1_20251222_143022_a3f9b2c1_metadata.json`

The metadata file contains:
- Original characteristics
- Generated prompt
- Timestamp
- Image path

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

Or use the full module path:
```bash
python -m src.main
```

## Cost Considerations

Google Cloud Vertex AI charges per image generation. Check the [Vertex AI pricing page](https://cloud.google.com/vertex-ai/pricing) for current rates. Monitor your usage in the Google Cloud Console.

## Development

### Adding New Features

- **Custom prompt templates**: Edit [src/prompt_generator.py](src/prompt_generator.py)
- **Different image formats**: Modify [src/image_saver.py](src/image_saver.py)
- **Batch processing**: Extend [src/main.py](src/main.py)

### Testing

Generate a single test image:
```python
from prompt_generator import generate_prompt
from main import generate_image_from_prompt

prompt = generate_prompt({"age": "young", "gender": "female", "expression": "smiling"})
print(prompt)
```

## License

This project is for educational and research purposes.

## Support

For issues related to:
- **Google Cloud**: Check [Google Cloud Documentation](https://cloud.google.com/docs)
- **Vertex AI**: See [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- **This pipeline**: Open an issue in the repository
