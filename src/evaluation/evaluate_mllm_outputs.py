#!/usr/bin/env python3
"""Evaluate MLLM judgement outputs under output/judgements/<model>.

Per model, this script creates output/evaluation/<model> with:
1) Aggregated probability scores per (base face, variation, scenario)
2) Paired comparison stats between base and variations:
   delta = score(variation, scenario) - score(base, scenario)
   including distribution metrics, histogram, t-test / Wilcoxon, and Cohen's d.

Usage examples:
  python3 src/evaluate_mllm_outputs.py --model-folder llave_next
  python3 src/evaluate_mllm_outputs.py --all-models
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


MAX_SCENARIO_INCLUDED = 25

HEATMAP_CMAP = "coolwarm"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_numbers_from_filename(file_name: str):
    scen_match = re.search(r"scenario(\d+)", file_name, flags=re.IGNORECASE)
    order_match = re.search(r"order(\d+)", file_name, flags=re.IGNORECASE)
    seed_match = re.search(r"seed(\d+)", file_name, flags=re.IGNORECASE)
    scenario = int(scen_match.group(1)) if scen_match else None
    order = int(order_match.group(1)) if order_match else None
    seed = int(seed_match.group(1)) if seed_match else None
    return scenario, order, seed


def _extract_choice(data: dict) -> str | None:
    chosen = data.get("chosen_option")
    if isinstance(chosen, str) and chosen.strip():
        return chosen.strip()

    raw_output = (data.get("raw_output") or "").strip()
    marker = re.search(r"\((a|b)\)", raw_output, flags=re.IGNORECASE)
    options = data.get("options")
    if marker and isinstance(options, dict):
        key = marker.group(1).lower()
        choice = options.get(key)
        if isinstance(choice, str) and choice.strip():
            return choice.strip()

    return None


def _extract_options(data: dict) -> tuple[list[str], str | None]:
    options = data.get("options")
    if isinstance(options, dict):
        if "a" in options and "b" in options:
            option_a = str(options["a"]).strip()
            option_b = str(options["b"]).strip()
            return [option_a, option_b], option_a
        values = [str(v).strip() for _, v in sorted(options.items())]
        return values, values[0] if values else None

    if isinstance(options, list):
        values = [str(v).strip() for v in options]
        return values, values[0] if values else None

    return [], None


def _is_base_variation(face_folder_name: str, variation_folder_name: str) -> bool:
    return variation_folder_name == face_folder_name


def _get_base_variation_folder(face_folder: Path) -> Path | None:
    subfolders = sorted([p for p in face_folder.iterdir() if p.is_dir()])
    if not subfolders:
        return None

    exact_match = next((p for p in subfolders if p.name == face_folder.name), None)
    if exact_match:
        return exact_match

    return subfolders[0]


def _extract_base_characteristics(base_folder: Path) -> dict:
    defaults = {"age": "unknown", "gender": "unknown", "ethnicity": "unknown", "body_index": "unknown"}
    json_files = sorted(base_folder.glob("*.json"))
    if not json_files:
        return defaults

    data = _read_json(json_files[0])
    if not data:
        return defaults

    characteristics = data.get("characteristics") or {}
    age = characteristics.get("age", defaults["age"])
    gender = characteristics.get("gender", defaults["gender"])
    ethnicity = characteristics.get("ethnicity", defaults["ethnicity"])
    body_index = characteristics.get("body_index")
    if body_index is None:
        body_index = characteristics.get("body_type", defaults["body_index"])

    return {
        "age": str(age),
        "gender": str(gender),
        "ethnicity": str(ethnicity),
        "body_index": str(body_index),
    }


def _save_json(path: Path, obj: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_rows_csv(path: Path, header: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _load_variation_whitelist(path: Path) -> dict[str, set[str]]:
    data = _read_json(path)
    if not isinstance(data, dict):
        return {}

    whitelist: dict[str, set[str]] = {}
    for key, values in data.items():
        if not isinstance(values, list):
            continue
        whitelist[str(key).strip().lower()] = {str(v).strip().lower() for v in values}
    return whitelist


VARIATION_WHITELIST = _load_variation_whitelist(Path("config/variation_features_whitelist.json"))


def _get_variation_value(characteristics: dict, key: str) -> str:
    if key in characteristics:
        return str(characteristics.get(key, "")).strip().lower()

    variation = characteristics.get("variation")
    if isinstance(variation, dict):
        return str(variation.get(key, "")).strip().lower()

    return ""


def _should_skip_variation(characteristics: dict, base_gender: str | None) -> bool:
    if base_gender:
        gender = str(base_gender).strip().lower()
    else:
        gender = str(characteristics.get("gender", "")).strip().lower()

    hair_style = _get_variation_value(characteristics, "hair_style")
    lip_makeup = _get_variation_value(characteristics, "lip_makeup_female")
    fashion_style = _get_variation_value(characteristics, "fashion_style")

    if gender == "male" and hair_style in {"braid", "bun"}:
        return True
    if lip_makeup == "neutral lipstick":
        return True
    if lip_makeup == "bold colors":
        return True
    if fashion_style == "daring / provocative":
        return True

    variation = characteristics.get("variation")
    if isinstance(variation, dict):
        for key, value in variation.items():
            normalized_key = str(key).strip().lower()
            normalized_value = str(value).strip().lower()
            allowed_values = VARIATION_WHITELIST.get(normalized_key)
            if allowed_values is None:
                return True
            if normalized_value not in allowed_values:
                return True

    return False


def _load_scenario_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    data = _read_json(path)
    if not isinstance(data, list):
        return labels

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        option_a = str(item.get("a", "")).strip()
        option_b = str(item.get("b", "")).strip()
        if option_a and option_b:
            labels[index] = f"{option_a} | {option_b}"
        elif option_a:
            labels[index] = option_a
        elif option_b:
            labels[index] = option_b
    return labels


def _load_scenario_options(path: Path) -> dict[int, tuple[str, str]]:
    options: dict[int, tuple[str, str]] = {}
    data = _read_json(path)
    if not isinstance(data, list):
        return options

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        option_a = str(item.get("a", "")).strip()
        option_b = str(item.get("b", "")).strip()
        if option_a and option_b:
            options[index] = (option_a, option_b)
    return options


def _probability_for_option(row: dict, target_option: str) -> float | None:
    if not target_option:
        return None

    counts = row.get("counts") or {}
    total = int(row.get("n") or 0)
    if total <= 0 or not isinstance(counts, dict):
        return None

    target_norm = _normalize_text(target_option)
    for option, count in counts.items():
        if _normalize_text(str(option)) == target_norm:
            return _safe_float(count, 0.0) / total

    # If the target option is part of the scenario options but received zero votes,
    # its probability is 0.0 rather than missing.
    for option in row.get("options") or []:
        if _normalize_text(str(option)) == target_norm:
            return 0.0
    return None


def _extract_variation_name(face_folder_name: str, variation_folder: Path) -> str:
    if variation_folder.name == face_folder_name:
        return "base"

    first_json = next(iter(sorted(variation_folder.glob("*.json"))), None)
    if first_json is not None:
        data = _read_json(first_json)
        if isinstance(data, dict):
            variation = (data.get("characteristics") or {}).get("variation")
            if isinstance(variation, dict) and variation:
                parts = [f"{k}:{v}" for k, v in sorted(variation.items())]
                return " | ".join(parts)

    if variation_folder.name.startswith(face_folder_name + "_"):
        return variation_folder.name[len(face_folder_name) + 1 :]
    return variation_folder.name


def _plot_variation_impact_heatmap(
    rows: list[dict],
    scenario_labels: dict[int, str],
    gender: str,
    output_file: Path,
):
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No variation-impact rows for {gender}", ha="center", va="center")
        fig.tight_layout()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=180)
        plt.close(fig)
        return

    def _parent_category(variation_name: str) -> str:
        name = str(variation_name or "").strip()
        first_part = name.split(" | ", 1)[0]
        if ":" in first_part:
            key = first_part.split(":", 1)[0].strip().lower()
        else:
            key = first_part.strip().lower()
        if key.endswith("_male"):
            key = key[: -len("_male")]
        if key.endswith("_female"):
            key = key[: -len("_female")]
        return key or "other"

    variations_unique = sorted({row["variation_name"] for row in rows})
    variations = sorted(variations_unique, key=lambda name: (_parent_category(name), str(name).lower()))
    parent_sequence = [_parent_category(name) for name in variations]
    scenarios = sorted({int(row["scenario"]) for row in rows})

    value_map = {(row["variation_name"], int(row["scenario"])): row["mean_delta"] for row in rows}
    matrix = []
    for variation in variations:
        matrix.append([value_map.get((variation, scenario), float("nan")) for scenario in scenarios])

    data = np.array(matrix, dtype=float)
    max_abs = np.nanmax(np.abs(data)) if np.size(data) else 0.0
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1e-6

    height = max(8, len(variations) * 0.35)
    width = max(10, len(scenarios) * 0.4)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(
        data,
        aspect="auto",
        cmap=HEATMAP_CMAP,
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
    )

    ax.set_yticks(range(len(variations)))
    ax.set_yticklabels(variations, fontsize=8)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([scenario_labels.get(s, f"Scenario {s}") for s in scenarios], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("variation")
    ax.set_xlabel("scenario options")
    ax.set_title(f"Variation Impact vs Base (delta p(option_a)) - {gender}")

    # Draw horizontal separator lines between parent variation categories.
    for idx in range(1, len(parent_sequence)):
        if parent_sequence[idx] != parent_sequence[idx - 1]:
            ax.axhline(y=idx - 0.5, color="black", linewidth=1.4)

    fig.colorbar(image, ax=ax, label="mean delta (variation - base)")
    fig.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def _plot_variation_impact_heatmap_combined(
    female_rows: list[dict],
    male_rows: list[dict],
    scenario_labels: dict[int, str],
    output_file: Path,
):
    """Single combined heatmap matching the style of the previous standalone script.

    Shared variations average male and female mean_delta; gender-specific
    variations (facial_hair_male, makeup_female, lip_makeup_female) use
    only the applicable gender's data.
    """
    FEMALE_ONLY_KEYS = {"makeup_female", "lip_makeup_female"}
    MALE_ONLY_KEYS   = {"facial_hair_male"}

    CAT_DISPLAY = {
        "accessories":         "Accessories",
        "eyewear":             "Eyewear",
        "facial_hair":         "Facial Hair",
        "fashion_style":       "Fashion",
        "hair_color":          "Hair Color",
        "hair_length":         "Hair Length",
        "hair_style":          "Hair Style",
        "lip_makeup":          "Lip Makeup",
        "makeup":              "Makeup",
        "piercings":           "Piercings",
        "skin_irregularities": "Skin",
        "tattoos":             "Tattoos",
    }

    def _parent(var: str) -> str:
        key = var.split(":", 1)[0].strip().lower() if ":" in var else var.strip().lower()
        if key.endswith("_male"):   key = key[: -len("_male")]
        if key.endswith("_female"): key = key[: -len("_female")]
        return key or "other"

    def _short(var: str) -> str:
        return var.split(":", 1)[1].strip() if ":" in var else var

    # Build per-(variation, scenario) accumulator, respecting gender-specific rules
    from collections import defaultdict
    acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in female_rows:
        cat = row["variation_name"].split(":", 1)[0].strip().lower()
        if cat not in MALE_ONLY_KEYS:
            acc[row["variation_name"]][int(row["scenario"])].append(float(row["mean_delta"]))
    for row in male_rows:
        cat = row["variation_name"].split(":", 1)[0].strip().lower()
        if cat not in FEMALE_ONLY_KEYS:
            acc[row["variation_name"]][int(row["scenario"])].append(float(row["mean_delta"]))

    if not acc:
        return

    scenarios  = sorted({s for v in acc.values() for s in v})
    variations = sorted(acc, key=lambda n: (_parent(n), n.lower()))
    parent_seq = [_parent(v) for v in variations]
    short_labels = [_short(v) for v in variations]

    # Category spans for bracket labels
    cat_spans: list[tuple[int, int, str]] = []
    start = 0
    for idx in range(1, len(parent_seq)):
        if parent_seq[idx] != parent_seq[idx - 1]:
            label = CAT_DISPLAY.get(parent_seq[start],
                                    parent_seq[start].replace("_", " ").title())
            cat_spans.append((start, idx - 1, label))
            start = idx
    cat_spans.append((start, len(parent_seq) - 1,
                      CAT_DISPLAY.get(parent_seq[start],
                                      parent_seq[start].replace("_", " ").title())))

    data = np.array(
        [[float(np.mean(acc[v][s])) if acc[v].get(s) else float("nan")
          for s in scenarios]
         for v in variations],
        dtype=float,
    )

    max_abs = float(np.nanmax(np.abs(data))) if data.size else 0.0
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1e-6

    FS     = 13
    CELL   = 0.42
    L, R, T, B = 2.2, 2.8, 0.9, 3.2
    CB_W, CB_GAP, CAT_W, CB_PAD = 0.25, 0.10, 1.10, 0.15

    n_vars, n_scen = len(variations), len(scenarios)
    axes_w = n_scen * CELL
    axes_h = n_vars  * CELL
    fig_w  = L + axes_w + R
    fig_h  = T + axes_h + B

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left   = L / fig_w,
        right  = (L + axes_w) / fig_w,
        bottom = B / fig_h,
        top    = (B + axes_h) / fig_h,
    )

    image = ax.imshow(
        data, aspect="auto", cmap=HEATMAP_CMAP,
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
    )

    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(short_labels, fontsize=FS)
    ax.set_xticks(range(n_scen))
    ax.set_xticklabels(
        [scenario_labels.get(s, f"Scenario {s}") for s in scenarios],
        rotation=70, ha="right", fontsize=FS,
    )
    ax.set_ylabel("variation", fontsize=FS)
    ax.set_xlabel("scenario options", fontsize=FS)
    ax.set_title("Variation Impact vs Base (delta p(option_a)) - combined", fontsize=FS)

    for idx in range(1, len(parent_seq)):
        if parent_seq[idx] != parent_seq[idx - 1]:
            ax.axhline(y=idx - 0.5, color="black", linewidth=1.4)

    cb_left = (L + axes_w + CB_GAP + CAT_W + CB_PAD) / fig_w
    cbar_ax = fig.add_axes([cb_left, B / fig_h, CB_W / fig_w, axes_h / fig_h])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=FS)
    cbar.set_label("mean delta (variation - base)", fontsize=FS)

    ax_right_fig = (L + axes_w) / fig_w
    for s, e, cat in cat_spans:
        y_norm = 1.0 - (((s + e) / 2) + 0.5) / n_vars
        y_fig  = B / fig_h + y_norm * (axes_h / fig_h)
        fig.text(ax_right_fig + CB_GAP / fig_w, y_fig, cat,
                 va="center", ha="left", fontsize=FS,
                 color="#333333", fontweight="bold")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_histogram(deltas: list[float], output_file: Path, title: str):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    if deltas:
        bins = min(40, max(10, int(math.sqrt(len(deltas)))))
        ax.hist(deltas, bins=bins)
    ax.axvline(0.0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Delta = p(option_a | variation) - p(option_a | base)")
    ax.set_ylabel("frequency")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def _cohens_d_paired(deltas: list[float]) -> float | None:
    n = len(deltas)
    if n < 2:
        return None
    mean_delta = sum(deltas) / n
    variance = sum((value - mean_delta) ** 2 for value in deltas) / (n - 1)
    sd = math.sqrt(variance)
    if sd == 0:
        return 0.0
    return mean_delta / sd


def _run_tests(deltas: list[float]) -> dict:
    result = {
        "n": len(deltas),
        "normality": None,
        "paired_t_test": None,
        "wilcoxon_signed_rank": None,
        "recommended_test": None,
    }
    if len(deltas) < 2:
        return result

    # Degenerate case: all deltas equal -> variance zero, infer no dispersion.
    if all(abs(value - deltas[0]) < 1e-12 for value in deltas):
        result["normality"] = {
            "test": "shapiro",
            "note": "skipped_constant_input",
            "roughly_normal": True,
        }
        result["paired_t_test"] = {
            "test": "paired_t_as_one_sample_on_delta",
            "note": "skipped_constant_input",
            "statistic": None,
            "p_value": 1.0 if abs(deltas[0]) < 1e-12 else 0.0,
        }
        result["wilcoxon_signed_rank"] = {
            "note": "skipped_constant_input",
            "statistic": None,
            "p_value": 1.0 if abs(deltas[0]) < 1e-12 else 0.0,
        }
        result["recommended_test"] = "paired_t_test"
        return result

    if scipy_stats is None:
        result["recommended_test"] = "scipy_not_available"
        return result

    roughly_normal = None
    if 3 <= len(deltas) <= 5000:
        try:
            shapiro_stat, shapiro_p = scipy_stats.shapiro(deltas)
            roughly_normal = shapiro_p > 0.05
            result["normality"] = {
                "test": "shapiro",
                "statistic": float(shapiro_stat),
                "p_value": float(shapiro_p),
                "roughly_normal": bool(roughly_normal),
            }
        except Exception:
            roughly_normal = None
            result["normality"] = {"test": "shapiro", "error": "failed"}

    try:
        t_stat, t_p = scipy_stats.ttest_1samp(deltas, popmean=0.0)
        result["paired_t_test"] = {
            "test": "paired_t_as_one_sample_on_delta",
            "statistic": float(t_stat),
            "p_value": float(t_p),
        }
    except Exception:
        result["paired_t_test"] = {"error": "failed"}

    try:
        wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(deltas, zero_method="wilcox")
        result["wilcoxon_signed_rank"] = {
            "statistic": float(wilcoxon_stat),
            "p_value": float(wilcoxon_p),
        }
    except Exception:
        result["wilcoxon_signed_rank"] = {"error": "failed"}

    if roughly_normal is True:
        result["recommended_test"] = "paired_t_test"
    else:
        result["recommended_test"] = "wilcoxon_signed_rank"

    return result


def evaluate_model(model_dir: Path, evaluation_dir: Path, max_faces: int = 0):
    if not model_dir.exists():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    scenario_options = _load_scenario_options(Path("config/judgement_scenarios.json"))

    probability_rows: list[dict] = []
    pair_rows: list[dict] = []

    faces_seen = 0
    for face_folder in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        if max_faces > 0 and faces_seen >= max_faces:
            break
        faces_seen += 1

        base_folder = _get_base_variation_folder(face_folder)
        if base_folder is None:
            continue

        base_chars = _extract_base_characteristics(base_folder)
        base_gender = _normalize_text(base_chars.get("gender", ""))
        grouped = defaultdict(lambda: {
            "face_folder": face_folder.name,
            "variation_folder": "",
            "variation_is_base": False,
            "scenario": None,
            "option_a": None,
            "options": set(),
            "counts": Counter(),
            "n": 0,
        })

        variation_folders = sorted(p for p in face_folder.iterdir() if p.is_dir())
        for variation_folder in variation_folders:
            variation_name = _extract_variation_name(face_folder.name, variation_folder)
            json_files = sorted(variation_folder.glob("*.json"))
            for json_file in json_files:
                data = _read_json(json_file)
                if not data:
                    continue

                characteristics = data.get("characteristics") or {}
                if not _is_base_variation(face_folder.name, variation_folder.name):
                    if _should_skip_variation(characteristics, base_gender):
                        continue

                scenario, order, seed = _extract_numbers_from_filename(json_file.name)
                if scenario is None:
                    scenario = data.get("scenario") or data.get("scenario_id")
                if order is None:
                    order = data.get("order")
                if seed is None:
                    seed = data.get("seed")

                if scenario is None:
                    continue
                scenario = int(scenario)
                if scenario < 1 or scenario > MAX_SCENARIO_INCLUDED:
                    continue

                options, option_a = _extract_options(data)
                chosen = _extract_choice(data)
                if not chosen:
                    continue

                canonical_choice = None
                chosen_norm = _normalize_text(chosen)
                for option in options:
                    if _normalize_text(option) == chosen_norm:
                        canonical_choice = option
                        break
                if canonical_choice is None:
                    canonical_choice = chosen

                group_key = (variation_folder.name, scenario)
                entry = grouped[group_key]
                entry["variation_folder"] = variation_folder.name
                entry["variation_name"] = variation_name
                entry["variation_is_base"] = _is_base_variation(face_folder.name, variation_folder.name)
                entry["scenario"] = scenario
                entry["option_a"] = option_a or entry["option_a"]
                entry["options"].update(options)
                entry["counts"][canonical_choice] += 1
                entry["n"] += 1

            # end json loop
        # end variation loop

        face_scenario_to_base = {}
        face_scenario_to_variations = defaultdict(list)

        for (_, scenario), entry in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            total = entry["n"]
            if total == 0:
                continue

            options_sorted = sorted([option for option in entry["options"] if option])
            option_a = entry["option_a"]
            if option_a is None and options_sorted:
                option_a = options_sorted[0]

            counts = dict(entry["counts"])
            p_option_a = _safe_float(counts.get(option_a, 0), 0.0) / total if option_a else None

            row = {
                "face_folder": face_folder.name,
                "variation_folder": entry["variation_folder"],
                "variation_name": entry.get("variation_name", entry["variation_folder"]),
                "variation_is_base": bool(entry["variation_is_base"]),
                "scenario": int(scenario),
                "n": total,
                "option_a": option_a,
                "p_option_a": p_option_a,
                "counts": counts,
                "options": options_sorted,
                "age": base_chars["age"],
                "gender": base_chars["gender"],
                "ethnicity": base_chars["ethnicity"],
                "body_index": base_chars["body_index"],
            }
            probability_rows.append(row)

            if row["variation_is_base"]:
                face_scenario_to_base[int(scenario)] = row
            else:
                face_scenario_to_variations[int(scenario)].append(row)

        for scenario, base_row in face_scenario_to_base.items():
            canonical_pair = scenario_options.get(int(scenario))
            if canonical_pair is not None:
                base_option_a, base_option_b = canonical_pair
                base_score = _probability_for_option(base_row, base_option_a)
            else:
                base_score = base_row.get("p_option_a")
                base_option_a = base_row.get("option_a")
                base_option_b = (
                    base_row["options"][0]
                    if len(base_row["options"]) == 2 and base_row["options"][1] == base_option_a
                    else (base_row["options"][1] if len(base_row["options"]) == 2 else None)
                )

            if base_score is None:
                continue

            for var_row in face_scenario_to_variations.get(scenario, []):
                if canonical_pair is not None:
                    var_score = _probability_for_option(var_row, base_option_a)
                else:
                    var_score = var_row.get("p_option_a")
                if var_score is None:
                    continue

                delta = var_score - base_score
                pair_rows.append({
                    "face_folder": face_folder.name,
                    "variation_folder": var_row["variation_folder"],
                    "variation_name": var_row["variation_name"],
                    "scenario": int(scenario),
                    "option_a": base_option_a,
                    "option_b": base_option_b,
                    "base_score": base_score,
                    "variation_score": var_score,
                    "delta": delta,
                    "age": base_chars["age"],
                    "gender": base_chars["gender"],
                    "ethnicity": base_chars["ethnicity"],
                    "body_index": base_chars["body_index"],
                })

    deltas_all = [row["delta"] for row in pair_rows]
    delta_mean = (sum(deltas_all) / len(deltas_all)) if deltas_all else None
    delta_std = None
    if len(deltas_all) >= 2:
        mean_value = delta_mean
        variance = sum((value - mean_value) ** 2 for value in deltas_all) / (len(deltas_all) - 1)
        delta_std = math.sqrt(variance)

    by_scenario = defaultdict(list)
    for row in pair_rows:
        by_scenario[row["scenario"]].append(row["delta"])

    per_scenario_stats = {}
    for scenario, scenario_deltas in sorted(by_scenario.items()):
        scenario_mean = sum(scenario_deltas) / len(scenario_deltas)
        scenario_std = None
        if len(scenario_deltas) >= 2:
            scenario_variance = sum((value - scenario_mean) ** 2 for value in scenario_deltas) / (len(scenario_deltas) - 1)
            scenario_std = math.sqrt(scenario_variance)
        per_scenario_stats[str(scenario)] = {
            "n": len(scenario_deltas),
            "mean_delta": scenario_mean,
            "std_delta": scenario_std,
            "cohens_d": _cohens_d_paired(scenario_deltas),
            "tests": _run_tests(scenario_deltas),
        }

    overall_stats = {
        "n_pairs": len(pair_rows),
        "mean_delta": delta_mean,
        "std_delta": delta_std,
        "cohens_d": _cohens_d_paired(deltas_all),
        "tests": _run_tests(deltas_all),
    }

    evaluation_dir.mkdir(parents=True, exist_ok=True)

    prob_rows_csv = []
    prob_rows_json = []
    for row in probability_rows:
        counts = row["counts"]
        options = row["options"]
        prob_map = {}
        for option in options:
            prob_map[option] = (counts.get(option, 0) / row["n"]) if row["n"] else 0.0

        out_row = {
            "face_folder": row["face_folder"],
            "variation_folder": row["variation_folder"],
            "variation_is_base": row["variation_is_base"],
            "scenario": row["scenario"],
            "n": row["n"],
            "option_a": row["option_a"],
            "p_option_a": row["p_option_a"],
            "probabilities": prob_map,
            "counts": counts,
            "base_characteristics": {
                "age": row["age"],
                "gender": row["gender"],
                "ethnicity": row["ethnicity"],
                "body_index": row["body_index"],
            },
        }
        prob_rows_json.append(out_row)

        # Flatten with option_a / option_b focus while keeping full JSON above.
        option_b = None
        if len(options) == 2 and row["option_a"] in options:
            option_b = options[0] if options[1] == row["option_a"] else options[1]

        p_option_b = None
        if option_b is not None:
            p_option_b = prob_map.get(option_b)

        prob_rows_csv.append([
            row["face_folder"],
            row["variation_folder"],
            int(row["variation_is_base"]),
            row["scenario"],
            row["n"],
            row["option_a"],
            row["p_option_a"],
            option_b,
            p_option_b,
            row["age"],
            row["gender"],
            row["ethnicity"],
            row["body_index"],
        ])

    _save_json(evaluation_dir / "probability_scores.json", prob_rows_json)
    _save_rows_csv(
        evaluation_dir / "probability_scores.csv",
        [
            "face_folder",
            "variation_folder",
            "variation_is_base",
            "scenario",
            "n",
            "option_a",
            "p_option_a",
            "option_b",
            "p_option_b",
            "age",
            "gender",
            "ethnicity",
            "body_index",
        ],
        prob_rows_csv,
    )

    _save_json(evaluation_dir / "paired_deltas.json", pair_rows)
    pair_csv_rows = [
        [
            row["face_folder"],
            row["variation_folder"],
            row["variation_name"],
            row["scenario"],
            row["option_a"],
            row["option_b"],
            row["base_score"],
            row["variation_score"],
            row["delta"],
            row["age"],
            row["gender"],
            row["ethnicity"],
            row["body_index"],
        ]
        for row in pair_rows
    ]
    _save_rows_csv(
        evaluation_dir / "paired_deltas.csv",
        [
            "face_folder",
            "variation_folder",
            "variation_name",
            "scenario",
            "option_a",
            "option_b",
            "base_score",
            "variation_score",
            "delta",
            "age",
            "gender",
            "ethnicity",
            "body_index",
        ],
        pair_csv_rows,
    )

    # Aggregate variation impact across all base faces for each scenario and split by gender.
    scenario_labels = _load_scenario_labels(Path("config/judgement_scenarios.json"))
    grouped_var_impact = defaultdict(list)
    for row in pair_rows:
        gender = _normalize_text(row.get("gender", ""))
        if gender not in {"male", "female"}:
            continue
        if int(row["scenario"]) < 1 or int(row["scenario"]) > MAX_SCENARIO_INCLUDED:
            continue
        key = (gender, row["variation_name"], int(row["scenario"]))
        grouped_var_impact[key].append(row["delta"])

    impact_summary_rows = []
    for (gender, variation_name, scenario), deltas in sorted(grouped_var_impact.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        mean_delta = sum(deltas) / len(deltas)
        std_delta = None
        if len(deltas) >= 2:
            mean_v = mean_delta
            variance = sum((value - mean_v) ** 2 for value in deltas) / (len(deltas) - 1)
            std_delta = math.sqrt(variance)
        impact_summary_rows.append({
            "gender": gender,
            "variation_name": variation_name,
            "scenario": int(scenario),
            "scenario_label": scenario_labels.get(int(scenario), f"Scenario {scenario}"),
            "n_pairs": len(deltas),
            "mean_delta": mean_delta,
            "std_delta": std_delta,
        })

    _save_json(evaluation_dir / "variation_impact_summary.json", impact_summary_rows)
    _save_rows_csv(
        evaluation_dir / "variation_impact_summary.csv",
        ["gender", "variation_name", "scenario", "scenario_label", "n_pairs", "mean_delta", "std_delta"],
        [
            [
                row["gender"],
                row["variation_name"],
                row["scenario"],
                row["scenario_label"],
                row["n_pairs"],
                row["mean_delta"],
                row["std_delta"],
            ]
            for row in impact_summary_rows
        ],
    )

    female_rows = [row for row in impact_summary_rows if row["gender"] == "female"]
    male_rows = [row for row in impact_summary_rows if row["gender"] == "male"]
    _plot_variation_impact_heatmap(
        rows=female_rows,
        scenario_labels=scenario_labels,
        gender="female",
        output_file=evaluation_dir / "variation_impact_heatmap_female.png",
    )
    _plot_variation_impact_heatmap(
        rows=male_rows,
        scenario_labels=scenario_labels,
        gender="male",
        output_file=evaluation_dir / "variation_impact_heatmap_male.png",
    )
    _plot_variation_impact_heatmap_combined(
        female_rows=female_rows,
        male_rows=male_rows,
        scenario_labels=scenario_labels,
        output_file=evaluation_dir / "variation_impact_heatmap_combined.png",
    )

    _plot_delta_histogram(
        deltas_all,
        evaluation_dir / "delta_histogram_overall.png",
        "Distribution of Delta Across All (base, variation, scenario) Pairs",
    )

    stats_payload = {
        "overall": overall_stats,
        "per_scenario": per_scenario_stats,
    }
    _save_json(evaluation_dir / "paired_delta_statistics.json", stats_payload)

    summary_lines = [
        f"model: {model_dir.name}",
        f"faces_processed: {faces_seen}",
        f"pairs_total: {overall_stats['n_pairs']}",
        f"variation_impact_rows: {len(impact_summary_rows)}",
        f"variation_impact_rows_female: {len(female_rows)}",
        f"variation_impact_rows_male: {len(male_rows)}",
        f"mean_delta: {overall_stats['mean_delta']}",
        f"std_delta: {overall_stats['std_delta']}",
        f"cohens_d: {overall_stats['cohens_d']}",
        f"recommended_test: {overall_stats['tests'].get('recommended_test')}",
    ]
    (evaluation_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def resolve_model_dirs(judgements_root: Path, model_folder: str | None, all_models: bool) -> list[Path]:
    if all_models:
        return sorted([p for p in judgements_root.iterdir() if p.is_dir()])

    selected = model_folder or "llava_next"
    aliases = {
        "llava_next": "llava_next",
    }
    selected = aliases.get(selected, selected)
    return [judgements_root / selected]


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLLM judgement outputs.")
    parser.add_argument("--model-folder", default=None, help="Folder name under output/judgements (e.g., llave_next, qwen3)")
    parser.add_argument("--all-models", action="store_true", help="Evaluate all model folders under output/judgements")
    parser.add_argument("--judgements-root", default="output/judgements", help="Root path with model judgement folders")
    parser.add_argument("--evaluation-root", default="output/evaluation", help="Root path for evaluation outputs")
    parser.add_argument("--max-faces", type=int, default=0, help="Optional limit for processed base faces (0 = all)")
    args = parser.parse_args()

    judgements_root = Path(args.judgements_root)
    evaluation_root = Path(args.evaluation_root)

    model_dirs = resolve_model_dirs(judgements_root, args.model_folder, args.all_models)
    if not model_dirs:
        raise FileNotFoundError(f"No model folders found under: {judgements_root}")

    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"[skip] missing model folder: {model_dir}")
            continue
        out_dir = evaluation_root / model_dir.name
        evaluate_model(model_dir=model_dir, evaluation_dir=out_dir, max_faces=args.max_faces)
        print(f"[ok] saved evaluation outputs to: {out_dir}")


if __name__ == "__main__":
    main()
