"""Summarize base-face demographic counts from the final dataset."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def find_base_metadata_json(folder: Path) -> Path | None:
    exact = folder / f"{folder.name}_metadata.json"
    if exact.exists():
        return exact

    json_files = list(folder.glob("*.json"))
    base_candidates = [
        p
        for p in json_files
        if p.name.endswith("_metadata.json")
        and "_face_" not in p.name
        and "_body_" not in p.name
    ]
    if len(base_candidates) == 1:
        return base_candidates[0]
    if len(base_candidates) > 1:
        base_candidates.sort()
        return base_candidates[0]
    return None


def main() -> None:
    dataset_dir = Path("output/final_dataset")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    subfolders = [p for p in dataset_dir.iterdir() if p.is_dir()]
    print(f"Base face folders: {len(subfolders)}")

    counts: Counter[tuple[str, str, str, str]] = Counter()
    age_counts: Counter[str] = Counter()
    gender_counts: Counter[str] = Counter()
    ethnicity_counts: Counter[str] = Counter()
    body_type_counts: Counter[str] = Counter()
    missing_folders: list[str] = []
    for folder in sorted(subfolders):
        meta_path = find_base_metadata_json(folder)
        if meta_path is None:
            missing_folders.append(folder.name)
            continue
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        characteristics = data.get("characteristics", {})
        age = characteristics.get("age", "")
        gender = characteristics.get("gender", "")
        ethnicity = characteristics.get("ethnicity", "")
        body_type = characteristics.get("body_type", "normal")
        counts[(age, gender, ethnicity, body_type)] += 1
        age_counts[age] += 1
        gender_counts[gender] += 1
        ethnicity_counts[ethnicity] += 1
        body_type_counts[body_type] += 1

    if missing_folders:
        print(f"Folders missing base metadata JSON: {len(missing_folders)}")
        for name in missing_folders:
            print(f"  - {name}")

    print("\nStatistics:")
    for (age, gender, ethnicity, body_type), count in counts.most_common():
        print(
            f"{count} times faces of (age: {age}, gender: {gender}, "
            f"ethnicity: {ethnicity}, body_type: {body_type})"
        )

    if counts:
        labels = [
            f"{age}, {gender}, {ethnicity}, {body_type}"
            for (age, gender, ethnicity, body_type), _ in counts.most_common()
        ]
        values = [count for _, count in counts.most_common()]

        fig_height = max(8, len(labels) * 0.35)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.barh(range(len(labels)), values)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Count")
        ax.set_title("Base Face Characteristics")
        ax.set_ylim(len(labels) - 0.5, -0.5)
        fig.tight_layout(pad=0.5)

        output_path = Path("output/final_dataset_stats.png")
        fig.savefig(output_path, dpi=200)
        print(f"\nSaved chart to: {output_path}")

    def save_single_count_chart(title: str, counter: Counter[str], filename: str) -> None:
        if not counter:
            return
        labels = [label if label else "(missing)" for label, _ in counter.most_common()]
        values = [count for _, count in counter.most_common()]
        fig_height = max(4, len(labels) * 0.35)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.barh(range(len(labels)), values)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Count")
        ax.set_title(title)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        fig.tight_layout(pad=0.5)
        output_path = Path(f"output/{filename}")
        fig.savefig(output_path, dpi=200)
        print(f"Saved chart to: {output_path}")

    save_single_count_chart("Age Counts", age_counts, "final_dataset_age_counts.png")
    save_single_count_chart("Gender Counts", gender_counts, "final_dataset_gender_counts.png")
    save_single_count_chart("Ethnicity Counts", ethnicity_counts, "final_dataset_ethnicity_counts.png")
    save_single_count_chart("Body Type Counts", body_type_counts, "final_dataset_body_type_counts.png")


if __name__ == "__main__":
    main()
