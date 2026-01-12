"""
Prompt Generator Module
Generates image prompts by inserting characteristics from JSON data.
"""

def generate_prompt(characteristics):
    """
    Generate an image prompt from a set of characteristics.
    
    Args:
        characteristics (dict): Dictionary containing face characteristics
        
    Returns:
        str: Generated prompt for image generation
    """
    # Extract characteristics
    age = characteristics.get('age', 'adult')
    gender = characteristics.get('gender', 'person')
    ethnicity = characteristics.get('ethnicity', '')    
    body_type = characteristics.get('body_type', 'normal')
    
    # Build simple, direct prompt
    prompt = (
        f"Photorealistic studio portrait of an average-looking {body_type} build "
        f"{age} {gender} person with {ethnicity} facial features. "
        "Front-facing, neutral expression, head-and-shoulders framing. "
        "Bare face with uncovered ears, no earings, no jewelry, "
        "no accessories, no glasses, no headwear. "
        "Plain white studio backdrop, soft even lighting, no dramatic shadows. "
        "Natural skin texture with visible pores, minor blemishes, slight facial asymmetry, "
        "real-world imperfections, non-model appearance, realistic proportions. "
        "Documentary-style photography, not fashion, not advertising."
    )
    
    return prompt
