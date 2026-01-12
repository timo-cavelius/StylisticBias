"""
Image Generation Pipeline
Main script for generating face images using Google Cloud Vertex AI.
"""

import os
import json
import base64
from pathlib import Path
from itertools import product
import re
import requests
from dotenv import load_dotenv
from google.auth import default
from google.auth.transport.requests import Request

from prompt_generator import generate_prompt
from image_saver import save_image, save_metadata


def load_characteristics(filepath):
    """Load characteristics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_combinations(characteristics_dict):
    """
    Generate all possible combinations from base characteristics.
    
    Args:
        characteristics_dict (dict): Dictionary with lists of values for each category
        
    Returns:
        list: List of characteristic dictionaries (all combinations)
    """
    # Get category names and their values
    categories = list(characteristics_dict.keys())
    values = [characteristics_dict[cat] for cat in categories]
    
    # Generate all combinations
    combinations = []
    for combo in product(*values):
        char_dict = {categories[i]: combo[i] for i in range(len(categories))}
        combinations.append(char_dict)
    
    return combinations


def get_access_token():
    """Get Google Cloud access token for API authentication."""
    credentials, project = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    credentials.refresh(Request())
    return credentials.token


def generate_image_from_prompt(prompt, project_id, location="us-central1", model_name="imagen-4.0-generate-001"):
    """
    Generate an image using Google Cloud Vertex AI REST API.
    
    Args:
        prompt (str): Text prompt for image generation
        project_id (str): Google Cloud project ID
        location (str): API location (default: us-central1)
        model_name (str): Model name to use
        
    Returns:
        bytes: Generated image data
    """
    # Get authentication token
    access_token = get_access_token()
    
    # Construct REST API endpoint
    endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_name}:predict"
    
    # Build request payload
    payload = {
        "instances": [
            {
                "prompt": prompt
            }
        ],
        "parameters": {
            "sampleCount": 1,
            "enhancePrompt": False,
            "aspectRatio": "1:1"
        }
    }
    
    # Set headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Make API request
    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    
    # Parse response and extract image
    result = response.json()
    predictions = result.get("predictions", [])
    
    if not predictions:
        raise ValueError("No image generated in response")
    
    # Decode base64 image
    image_base64 = predictions[0].get("bytesBase64Encoded")
    if not image_base64:
        raise ValueError("No image data in response")
    
    image_bytes = base64.b64decode(image_base64)
    
    return image_bytes


def process_characteristics(characteristics_list, output_dir):
    """
    Process a list of characteristics and generate images.
    
    Args:
        characteristics_list (list): List of characteristic dictionaries
        output_dir (str): Directory to save generated images
        
    Returns:
        list: Paths to generated images
    """
    generated_images = []

    def get_next_face_index(dir_path: str) -> int:
        """Determine the next face index by scanning existing files in the output folder."""
        try:
            entries = os.listdir(dir_path)
        except FileNotFoundError:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            entries = []

        max_idx = 0
        for name in entries:
            # Match filenames starting with face_<number>_
            m = re.match(r"^face_(\d+)_", name)
            if m:
                try:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    continue
        return max_idx + 1
    
    start_index = get_next_face_index(output_dir)
    print(f"Starting face index: {start_index}")

    # Get project_id and location for REST API
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')

    for idx, characteristics in enumerate(characteristics_list, start_index):
        print(f"\n[{idx}/{len(characteristics_list)}] Processing characteristics...")
        
        # Generate prompt
        prompt = generate_prompt(characteristics)
        print(f"Prompt: {prompt}")
        
        try:
            # Generate image
            print("Generating image...")
            image_data = generate_image_from_prompt(prompt, project_id, location)
            
            # Save image with unique filename
            image_path = save_image(image_data, output_dir, prefix=f"face_{idx}")
            print(f"✓ Saved: {image_path}")
            
            # Save metadata
            metadata_path = save_metadata(image_path, characteristics, prompt)
            print(f"✓ Metadata: {metadata_path}")
            
            generated_images.append(image_path)
            
        except Exception as e:
            print(f"✗ Error generating image: {str(e)}")
            continue
    
    return generated_images


def main():
    """Main pipeline execution."""
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    output_dir = os.getenv('OUTPUT_DIR', 'output/images')
    characteristics_file = os.getenv('CHARACTERISTICS_FILE', 'config/characteristics.json')
    max_images = int(os.getenv('MAX_IMAGES', '0'))  # 0 = generate all
    
    # Validate configuration
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT_ID not set in .env file")
    
    print("=" * 60)
    print("Image Generation Pipeline")
    print("Using Vertex AI REST API")
    print("=" * 60)
    
    # Load characteristics
    print(f"\nLoading characteristics from: {characteristics_file}")
    characteristics_dict = load_characteristics(characteristics_file)
    
    # Generate all combinations
    print(f"Generating combinations...")
    characteristics_list = generate_combinations(characteristics_dict)
    print(f"Generated {len(characteristics_list)} combinations")
    print(f"  - Age options: {len(characteristics_dict.get('age', []))}")
    print(f"  - Gender options: {len(characteristics_dict.get('gender', []))}")
    print(f"  - Ethnicity options: {len(characteristics_dict.get('ethnicity', []))}")
    
    # Limit number of images if MAX_IMAGES is set
    if max_images > 0 and max_images < len(characteristics_list):
        characteristics_list = characteristics_list[:max_images]
        print(f"\n⚠ Limiting to first {max_images} images (set MAX_IMAGES=0 in .env to generate all)")
    
    # Process and generate images
    generated_images = process_characteristics(characteristics_list, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✓ Pipeline Complete!")
    print(f"Generated {len(generated_images)} images")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
