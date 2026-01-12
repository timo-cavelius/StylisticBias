# Image Generation Pipeline - Project Setup

This workspace contains an image generation pipeline using Google Cloud API.

## Project Overview
- Python-based pipeline for generating face images
- Reads characteristics from JSON file
- Generates prompts and creates images via Google Cloud API
- Saves images with unique filenames

## Setup Progress

- [x] Create copilot-instructions.md file
- [x] Scaffold Python project structure
- [x] Create configuration files
- [x] Implement main pipeline script
- [x] Create README documentation
- [x] Configure Python environment

## Project Structure
```
Image_generation/
├── src/
│   ├── main.py              # Main pipeline script
│   ├── prompt_generator.py  # Prompt generation logic
│   └── image_saver.py       # Image saving with unique names
├── output/
│   └── images/              # Generated images folder
├── config/
│   └── characteristics.json # Base characteristics data
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
└── README.md                # Setup instructions
```

## Development Guidelines
- Use Google Cloud Vertex AI API for image generation
- Always generate unique filenames (UUID or timestamp-based)
- Load characteristics from JSON config
- Handle API errors gracefully
- Log generation progress
