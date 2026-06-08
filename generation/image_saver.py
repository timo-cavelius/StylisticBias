"""
Image Saver Module
Handles saving images with unique filenames to prevent overwriting.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path


def generate_unique_filename(output_dir, prefix="image", extension="png"):
    """
    Generate a unique filename using UUID to prevent overwriting.
    
    Args:
        output_dir (str): Directory where images will be saved
        prefix (str): Prefix for the filename
        extension (str): File extension without dot
        
    Returns:
        str: Full path to the unique filename
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename using UUID
    unique_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}_{unique_id}.{extension}"
    
    return os.path.join(output_dir, filename)


def save_image(image_data, output_dir, prefix="generated"):
    """
    Save image data to a file with a unique filename.
    
    Args:
        image_data: Image data (bytes or PIL Image)
        output_dir (str): Directory where images will be saved
        prefix (str): Prefix for the filename
        
    Returns:
        str: Path to the saved image
    """
    filepath = generate_unique_filename(output_dir, prefix=prefix)
    
    # Handle different image data types
    if isinstance(image_data, bytes):
        # Save raw bytes
        with open(filepath, 'wb') as f:
            f.write(image_data)
    else:
        # Assume PIL Image or similar
        image_data.save(filepath)
    
    return filepath


def save_metadata(image_path, characteristics, prompt):
    """
    Save metadata about the generated image.
    
    Args:
        image_path (str): Path to the saved image
        characteristics (dict): Original characteristics used
        prompt (str): Generated prompt used
    """
    import json
    
    metadata_path = image_path.rsplit('.', 1)[0] + '_metadata.json'
    
    metadata = {
        'image_path': image_path,
        'characteristics': characteristics,
        'prompt': prompt,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path
