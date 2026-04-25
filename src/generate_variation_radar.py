#!/usr/bin/env python3
"""Generate variation category radar chart from existing comparison data."""

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def find_latest_comparison_folder(evaluation_root: Path) -> Optional[Path]:
    """Find the latest model_comparison folder."""
    comparison_dirs = sorted(evaluation_root.glob("model_comparison_*"))
    return comparison_dirs[-1] if comparison_dirs else None

def load_variation_data(csv_path: Path):
    """Load variation category data from CSV."""
    categories_metadata = [
        "skin_irregularities", "hair_color", "hair_length", "hair_style",
        "facial_hair_male", "makeup_female", "lip_makeup_female",
        "tattoos", "fashion_style", "eyewear", "piercings", "accessories"
    ]
    
    display_names = [
        "Skin irreg.", "Hair color", "Hair length", "Hair style",
        "Facial hair (M)", "Makeup (F)", "Lip makeup (F)",
        "Tattoos", "Fashion style", "Eyewear", "Piercings", "Accessories"
    ]
    
    # Identify models from the first row of CSV if possible, or use defaults
    models = ["gemma3", "gemma4", "llava_next", "pixtral", "qwen3"]
    colors = {
        "gemma3": "#E05A3A",
        "gemma4": "#5B8DD9",
        "llava_next": "#3DB89A",
        "pixtral": "#C07BD4",
        "qwen3": "#D4933A",
    }
    
    data = {model: {} for model in models}
    
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or len(row) < 2:
                continue
            category = row[0].strip()
            # If the row has 6 elements (cat + 5 models), it's the data we want
            if len(row) == 6:
                try:
                    values = [float(v) for v in row[1:6]]
                    for i, model in enumerate(models):
                        data[model][category] = values[i]
                except ValueError:
                    # Likely a header row or malformed data
                    continue
    
    model_data = {}
    for model in models:
        model_data[model] = [data[model].get(cat, 0.0) for cat in categories_metadata]
    
    return display_names, model_data, colors

def create_radar_chart(categories, data, colors, output_path):
    """Create and save the radar chart."""
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    
    max_val = 0.25
    ring_vals = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_ylim(0, max_val)
    ax.set_yticks(ring_vals)
    ax.set_yticklabels([f"{v:.2f}" for v in ring_vals], fontsize=9, color="#aaa")
    
    ax.grid(color="#ddd", linestyle="--", linewidth=0.7)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=11, color="#555")
    
    for model, values in data.items():
        vals_closed = values + [values[0]]
        ax.plot(angles_closed, vals_closed, color=colors[model], linewidth=2, label=model)
        ax.fill(angles_closed, vals_closed, color=colors[model], alpha=0.1)
    
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Radar chart saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, default=Path("output/evaluation"))
    parser.add_argument("--output", type=Path, default=Path("output/variation_radar_chart.png"))
    args = parser.parse_args()
    
    comp_folder = find_latest_comparison_folder(args.evaluation_root)
    if not comp_folder:
        print("No comparison folder found.")
        return
        
    csv_path = comp_folder / "variation_category_strength.csv"
    if not csv_path.exists():
        print(f"File {csv_path} not found.")
        return
        
    cats, data, colors = load_variation_data(csv_path)
    create_radar_chart(cats, data, colors, args.output)

if __name__ == "__main__":
    main()
