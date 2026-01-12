"""
Second pipeline: generate variations from base face images using Gemini image models.
- Reads base faces and metadata from a folder
- Builds gender-aware prompts with feature variations from config/variation_features.json
- Generates both face-focused and full-body outputs per variation
- Saves results under output/banana/<base_face_stem>/
"""

import json
import os
import shutil
import time
from itertools import product
from mimetypes import guess_type
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from image_saver import save_image, save_metadata


# ---------- Config & helpers ----------


def load_features(path: str) -> Dict[str, List[str]]:
    with open(path, "r") as f:
        return json.load(f)


def load_base_faces(base_dir: str) -> List[Tuple[Path, Optional[dict]]]:
    """Return list of (image_path, metadata_dict or None)."""
    img_exts = {".png", ".jpg", ".jpeg"}
    faces = []
    for entry in sorted(Path(base_dir).glob("**/*")):
        if entry.suffix.lower() not in img_exts:
            continue
        meta_path = entry.with_name(entry.stem + "_metadata.json")
        metadata = None
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
            except Exception:
                metadata = None
        faces.append((entry, metadata))
    return faces


def infer_gender(metadata: Optional[dict]) -> str:
    if not metadata:
        return "unknown"

    # Accept top-level gender/sex or nested under "characteristics".
    candidates = [metadata]
    if isinstance(metadata.get("characteristics"), dict):
        candidates.append(metadata["characteristics"])

    for scope in candidates:
        for key in ["gender", "sex"]:
            val = scope.get(key)
            if isinstance(val, str):
                low = val.lower()
                if "female" in low:
                    return "female"
                if "male" in low:
                    return "male"
    return "unknown"


def make_mime_type(path: Path) -> str:
    mime, _ = guess_type(str(path))
    if mime:
        return mime
    return "image/png"


def make_genai_client(project_id: str, location: str) -> genai.Client:
    # Uses Application Default Credentials (service account) and routes to Vertex AI.
    return genai.Client(vertexai=True, project=project_id, location=location)


# ---------- Prompt builders ----------


def build_face_prompt(base_gender: str, base_features: dict, variation: dict) -> str:
    gender_text = base_gender if base_gender != "unknown" else "person"
    parts = [
        f"A portrait photograph of the same {gender_text} as the reference image, facing the camera, neutral expression.",
        "Plain white background, even studio lighting.",
    ]
    # Face features
    for key in ["skin_irregularities", "hair_color", "hair_length", "hair_style", "eyewear", "piercings", "tattoos"]:
        val = variation.get(key)
        if val:
            parts.append(f"{key.replace('_', ' ')}: {val}.")
    if base_gender == "male":
        fh = variation.get("facial_hair_male")
        if fh:
            parts.append(f"Facial hair: {fh}.")
    if base_gender == "female":
        mk = variation.get("makeup_female")
        lip = variation.get("lip_makeup_female")
        if mk:
            parts.append(f"Makeup: {mk}.")
        if lip:
            parts.append(f"Lip makeup: {lip}.")
    accessories = variation.get("accessories")
    if accessories:
        parts.append(f"Accessories: {accessories}.")
    parts.append("Keep the face identity consistent with the reference image.")
    return " ".join(parts)


def build_body_prompt(base_gender: str, base_features: dict, variation: dict) -> str:
    gender_text = base_gender if base_gender != "unknown" else "person"
    style = variation.get("fashion_style", "casual")
    parts = [
        f"Generate a full-body portrait photograph of a {gender_text}, standing in neutral pose, facing camera.",
        f"Wearing {style} clothing/outfit.",
        "Same face and facial features as the reference image to maintain identity consistency.",
        "Plain white background, even studio lighting, professional photography style.",
        "Show the complete body from head to feet wearing the specified fashion style.",
    ]
    parts.append("Keep the face identity consistent with the reference image.")
    return " ".join(parts)

# ---------- Variation generation ----------


def iter_variations(features: dict, gender: str, limit: int, feature_filter: Optional[str] = None):
    """Yield variation dicts with single-feature changes from base.
    If feature_filter is provided, only yield variations for that feature.
    """
    common_keys = [
        "skin_irregularities",
        "hair_color",
        "hair_length",
        "hair_style",
        "eyewear",
        "piercings",
        "tattoos",
        "accessories",
        "fashion_style",
    ]

    if gender == "male":
        gender_keys = ["facial_hair_male"]
    elif gender == "female":
        gender_keys = ["makeup_female", "lip_makeup_female"]
    else:
        gender_keys = []

    all_keys = common_keys + gender_keys

    count = 0
    # Iterate through each feature key and each option for that feature
    for key in all_keys:
        # Skip if feature_filter is set and this key doesn't match
        if feature_filter and key != feature_filter:
            continue
            
        options = features.get(key, [])
        if not options:
            continue
        for option in options:
            if not option:  # Skip empty strings
                continue
            # Yield a variation with only this one feature changed
            variation = {key: option}
            yield variation
            count += 1
            if limit and count >= limit:
                return


# ---------- Model call ----------


def call_model(client: genai.Client, model_name: str, prompt: str, image_bytes: bytes, mime_type: str) -> bytes:
    if not prompt or not isinstance(prompt, str):
        raise ValueError(f"Invalid prompt: {repr(prompt)}")
    
    attempt = 0
    while True:
        try:
            # Recreate image_part on each attempt to avoid state issues
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, image_part],
            )

            if not response or not response.parts:
                raise ValueError("Empty response or no parts returned")

            for part in response.parts:
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data

            raise ValueError("No image data in response")
        except Exception as e:
            msg = str(e)
            retryable = ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("NoneType" in msg)
            if retryable:
                sleep_s = min(2 ** attempt, 60)  # Cap at 60 seconds
                print(f"Rate limited or empty response. Retrying in {sleep_s}s (attempt {attempt + 1})...")
                time.sleep(sleep_s)
                attempt += 1
                continue
            raise


# ---------- Pipeline ----------


def run_banana_pipeline():
    load_dotenv()

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")  # Regional endpoint required for Gemini image models
    base_faces_dir = os.getenv("BANANA_BASE_FACES_DIR", "base_faces")
    output_root = Path(os.getenv("BANANA_OUTPUT_ROOT", "output/banana"))
    features_path = os.getenv("BANANA_FEATURES_FILE", "config/variation_features.json")
    model_name = os.getenv("BANANA_MODEL_ID", "gemini-2.5-flash-image")
    raw_max_variations = (
        os.getenv("BANANA_MAX_VARIATIONS")
        or os.getenv("MAX_VARIATIONS")
        or "999"  # High default to process all features
    )
    max_variations = int(raw_max_variations)

    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT_ID is required")

    features = load_features(features_path)
    base_faces = load_base_faces(base_faces_dir)
    if not base_faces:
        print(f"No base faces found in {base_faces_dir}")
        return []

    print(f"Loaded {len(base_faces)} base faces from {base_faces_dir}")
    print(f"Max variations per face: {max_variations}")

    test_feature = os.getenv("TEST_FEATURE")  # Optional: filter to single feature for testing
    if test_feature:
        print(f"TEST MODE: Only generating variations for '{test_feature}' feature")

    client = make_genai_client(project_id, location)

    all_outputs = []

    for face_idx, (img_path, metadata) in enumerate(base_faces, 1):
        print(f"\n[{face_idx}/{len(base_faces)}] Processing {img_path.name}")
        gender = infer_gender(metadata)
        print(f"Detected gender: {gender}")

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        mime_type = make_mime_type(img_path)

        dest_dir = output_root / img_path.stem
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Move original image and metadata from base_faces to destination folder
        orig_dest = dest_dir / img_path.name
        shutil.move(str(img_path), str(orig_dest))
        all_outputs.append(orig_dest)
        print(f"✓ Moved original image to {orig_dest}")
        
        meta_src = img_path.with_name(img_path.stem + "_metadata.json")
        if meta_src.exists():
            meta_dest = dest_dir / (img_path.stem + "_metadata.json")
            shutil.move(str(meta_src), str(meta_dest))
            print(f"✓ Moved original metadata to {meta_dest}")

        variation_iter = iter_variations(features, gender, max_variations, feature_filter=test_feature)
        for var_idx, variation in enumerate(variation_iter, 1):
            is_fashion_variation = "fashion_style" in variation

            var_attempt = 0
            var_success = False
            while not var_success and var_attempt < 5:
                try:
                    if is_fashion_variation:
                        # For fashion variations: generate ONLY body image
                        body_prompt = build_body_prompt(gender, metadata or {}, variation)
                        body_img = call_model(client, model_name, body_prompt, img_bytes, mime_type)
                        body_path = save_image(body_img, dest_dir, prefix=f"{img_path.stem}_body_{var_idx}")
                        save_metadata(body_path, {"base": img_path.name, "variation": variation, "prompt": body_prompt}, body_prompt)
                        all_outputs.append(body_path)
                        print(f"✓ Variation {var_idx}: saved body image (fashion style)")
                    else:
                        # For other features: generate ONLY face image
                        face_prompt = build_face_prompt(gender, metadata or {}, variation)
                        face_img = call_model(client, model_name, face_prompt, img_bytes, mime_type)
                        face_path = save_image(face_img, dest_dir, prefix=f"{img_path.stem}_face_{var_idx}")
                        save_metadata(face_path, {"base": img_path.name, "variation": variation, "prompt": face_prompt}, face_prompt)
                        all_outputs.append(face_path)
                        print(f"✓ Variation {var_idx}: saved face image")

                    var_success = True
                except Exception as e:
                    var_attempt += 1
                    print(f"✗ Variation {var_idx} attempt {var_attempt} failed: {e}")
                    if var_attempt >= 5:
                        print(f"Skipping variation {var_idx} after 5 failed attempts.")
                        break
                    sleep_s = min(2 ** (var_attempt - 1), 60)
                    print(f"Retrying variation {var_idx} in {sleep_s}s...")
                    time.sleep(sleep_s)

    print("\nBanana pipeline complete.")
    return all_outputs


if __name__ == "__main__":
    run_banana_pipeline()
