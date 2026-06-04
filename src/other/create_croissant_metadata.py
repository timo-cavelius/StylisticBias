"""Create Croissant metadata for the Kaggle dataset (JSON-only schema).

Usage:
    python src/create_croissant_metadata.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlcroissant as mlc


ROOT_DIR = Path(__file__).resolve().parents[2]
HF_DIR = ROOT_DIR / "output" / "hf_dataset"
CROISSANT_PATH = HF_DIR / "croissant.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_croissant() -> dict:
    images = mlc.FileSet(
        id="files-generated-images",
        name="generated_images",
        description="PNG image files in the dataset.",
        includes=["output/final_dataset/**/*.png"],
        encoding_formats=["image/png"],
    )

    metadata_json_files = mlc.FileSet(
        id="files-metadata-json",
        name="metadata_json",
        description="Per-image JSON metadata files containing prompts and variation attributes.",
        includes=["output/final_dataset/**/*_metadata.json"],
        encoding_formats=["application/json"],
    )

    fields = [
        mlc.Field(
            id="field-image-path",
            name="image_path",
            description="Path to the corresponding image file.",
            data_types=[mlc.DataType.URL],
            source=mlc.Source(file_set="files-metadata-json", extract=mlc.Extract(json_path="$.image_path")),
        ),
        mlc.Field(
            id="field-base-image",
            name="base_image",
            description="Base identity image filename used for variation generation.",
            data_types=[mlc.DataType.TEXT],
            source=mlc.Source(
                file_set="files-metadata-json", extract=mlc.Extract(json_path="$.characteristics.base")
            ),
        ),
        mlc.Field(
            id="field-prompt",
            name="prompt",
            description="Prompt used to synthesize the image variation.",
            data_types=[mlc.DataType.TEXT],
            source=mlc.Source(file_set="files-metadata-json", extract=mlc.Extract(json_path="$.prompt")),
        ),
        mlc.Field(
            id="field-fashion-style",
            name="fashion_style",
            description="Optional fashion style variation label.",
            data_types=[mlc.DataType.TEXT],
            source=mlc.Source(
                file_set="files-metadata-json",
                extract=mlc.Extract(json_path="$.characteristics.variation.fashion_style"),
            ),
        ),
        mlc.Field(
            id="field-timestamp",
            name="timestamp",
            description="Timestamp when the variation image metadata was created.",
            data_types=[mlc.DataType.TEXT],
            source=mlc.Source(file_set="files-metadata-json", extract=mlc.Extract(json_path="$.timestamp")),
        ),
    ]

    record_set = mlc.RecordSet(
        id="recordset-generated-faces",
        name="generated_faces",
        description="Row-level metadata extracted from per-image JSON metadata files.",
        fields=fields,
    )

    metadata = mlc.Metadata(
        id="https://www.kaggle.com/datasets/stylistic-bias/diverse-face-variations",
        name="Diverse Face Variations",
        description=(
            "Synthetic face dataset with base and variation images, with optional "
            "attributes such as age, gender, ethnicity, and body type."
        ),
        url="https://www.kaggle.com/datasets/stylistic-bias/diverse-face-variations",
        license=["https://creativecommons.org/licenses/by/4.0/"],
        distribution=[images, metadata_json_files],
        record_sets=[record_set],
        keywords=["synthetic faces", "image generation", "kaggle"],
    )

    metadata_json = metadata.to_json()

    metadata_json["datePublished"] = "2026-05-05"
    metadata_json["version"] = "1.0"

    metadata_json["rai:dataLimitations"] = (
        "Dataset contains 500 base synthetic faces with limited attribute diversity. "
        "Gender distribution is imbalanced (274 male, 226 female). Age skews toward young adults (260), "
        "with 124 middle-aged and 116 elderly. Body type distribution: 186 normal build, 160 obese, 154 thin. "
        "Ethnicity distribution approximately balanced (110 Asian, 109 African, 101 European, 95 Middle Eastern, 85 Latino). "
        "Limited to these specific demographic attributes only. NOT RECOMMENDED FOR: Real-world identity verification, "
        "production face recognition systems, or as a representation of actual population demographics."
    )
    metadata_json["rai:dataBiases"] = (
        "Potential biases in synthetic generation process despite diversity efforts. Gender bias toward male representation (54.8%). "
        "Age bias toward younger adults. Body type and ethnicity distributions are not perfectly balanced. Selection bias inherent "
        "to synthetic data generation methodology."
    )
    metadata_json["rai:personalSensitiveInformation"] = [
        "Gender (binary: male/female)",
        "Age (categorical: young adult, middle-aged, elderly)",
        "Ethnicity (categorical: Asian, African, European, Middle Eastern, Latino)",
        "Body type (categorical: normal, obese, thin)",
    ]
    metadata_json["rai:hasSyntheticData"] = True
    metadata_json["rai:dataUseCases"] = (
        "VALIDITY ESTABLISHED FOR: Bias investigation in face attributes, fairness auditing of machine learning models, "
        "evaluation of multimodal large language models (MLLMs) on demographic perception tasks. NOT ESTABLISHED FOR: "
        "Real-world face recognition, identity verification, or production deployment."
    )
    metadata_json["rai:dataSocialImpact"] = (
        "NEGATIVE IMPACTS: Synthetic data may not fully represent real-world human diversity and may propagate bias if misused. "
        "POSITIVE IMPACTS: Avoids use of real identities and supports controlled fairness research."
    )
    metadata_json["prov:wasDerivedFrom"] = ["Google Imagen 4", "Nano Banana image generation service"]
    metadata_json["prov:wasGeneratedBy"] = (
        "Synthetic faces generated from prompts with demographic attributes (age, gender, ethnicity, body type) using Imagen 4 "
        "and Nano Banana. Prompts and pipeline details are available in src/prompt_generator.py and related generation scripts."
    )

    # Validate the emitted JSON-LD before writing it.
    mlc.Dataset(jsonld=metadata_json, debug=False)
    return metadata_json


def main() -> None:
    metadata_json = build_croissant()
    HF_DIR.mkdir(parents=True, exist_ok=True)
    CROISSANT_PATH.write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")
    print(f"Created Croissant metadata: {CROISSANT_PATH}")


if __name__ == "__main__":
    main()
