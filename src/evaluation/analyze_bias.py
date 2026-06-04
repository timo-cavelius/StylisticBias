#!/usr/bin/env python3
"""Analyze bias strength per scenario and per variation.

Moved from root `src/analyze_bias.py`.
"""
from __future__ import annotations

from pathlib import Path
import csv
import argparse
from collections import defaultdict
import statistics


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def analyze_scenarios(scenario_rows):
    agg = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for r in scenario_rows:
        try:
            scen = int(r.get("scenario"))
        except Exception:
            continue
        opt = r.get("option") or ""
        try:
            cnt = int(r.get("count") or 0)
        except Exception:
            try:
                total = int(r.get("total") or 0)
                prop = float(r.get("proportion") or 0.0)
                cnt = int(round(prop * total))
            except Exception:
                cnt = 0
        agg[scen][opt] += cnt
        totals[scen] += cnt

    out_rows = []
    for scen in sorted(agg.keys()):
        opts = agg[scen]
        total = totals.get(scen, 0)
        if total <= 0:
            continue
        top_opt, top_cnt = max(opts.items(), key=lambda x: x[1])
        top_prop = top_cnt / total
        bias = abs(top_prop - 0.5)
        out_rows.append({
            "scenario": scen,
            "top_option": top_opt,
            "top_count": top_cnt,
            "total": total,
            "top_proportion": f"{top_prop:.4f}",
            "bias": f"{bias:.4f}",
        })

    out_rows.sort(key=lambda r: float(r["bias"]))
    return out_rows


def analyze_variations(variation_rows):
    per_var_scen = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    per_var_totals = defaultdict(lambda: defaultdict(int))

    for r in variation_rows:
        var = r.get("variation") or r.get("variation", "")
        try:
            scen = int(r.get("scenario"))
        except Exception:
            continue
        opt = r.get("option") or ""
        try:
            cnt = int(r.get("count") or 0)
        except Exception:
            try:
                total = int(r.get("total") or 0)
                prop = float(r.get("proportion") or 0.0)
                cnt = int(round(prop * total))
            except Exception:
                cnt = 0
        per_var_scen[var][scen][opt] += cnt
        per_var_totals[var][scen] += cnt

    var_rows = []
    for var, scen_map in sorted(per_var_scen.items()):
        biases = []
        for scen, opts in scen_map.items():
            total = per_var_totals[var].get(scen, 0)
            if total <= 0:
                continue
            top_cnt = max(opts.values())
            top_prop = top_cnt / total
            bias = abs(top_prop - 0.5)
            biases.append(bias)
        if not biases:
            continue
        mean_bias = statistics.mean(biases)
        median_bias = statistics.median(biases)
        min_bias = min(biases)
        max_bias = max(biases)
        var_rows.append({
            "variation": var,
            "mean_bias": f"{mean_bias:.4f}",
            "median_bias": f"{median_bias:.4f}",
            "min_bias": f"{min_bias:.4f}",
            "max_bias": f"{max_bias:.4f}",
            "scenarios_count": len(biases),
        })

    var_rows.sort(key=lambda r: float(r["mean_bias"]))
    return var_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", default="output/evaluation/llave_next")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    eval_dir = Path(args.evaluation_dir)
    scen_csv = eval_dir / "scenario_choices.csv"
    var_csv = eval_dir / "variation_choices.csv"

    if not scen_csv.exists() and not var_csv.exists():
        print("No input CSVs found in:", eval_dir)
        return

    scenario_rows = read_csv(scen_csv) if scen_csv.exists() else []
    variation_rows = read_csv(var_csv) if var_csv.exists() else []

    scenario_biases = analyze_scenarios(scenario_rows)
    variation_biases = analyze_variations(variation_rows)

    write_csv(eval_dir / "scenario_biases.csv",
              ["scenario", "top_option", "top_count", "total", "top_proportion", "bias"],
              scenario_biases)

    write_csv(eval_dir / "variation_biases.csv",
              ["variation", "mean_bias", "median_bias", "min_bias", "max_bias", "scenarios_count"],
              variation_biases)

    top_n = args.top_n
    with (eval_dir / "bias_summary.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"Weakest {top_n} scenarios (closest to 0.5 majority):\n")
        for r in scenario_biases[:top_n]:
            fh.write(f"Scenario {r['scenario']}: bias={r['bias']} top={r['top_option']} ({r['top_proportion']})\n")
        fh.write("\n")
        fh.write(f"Weakest {top_n} variations (mean bias across scenarios):\n")
        for r in variation_biases[:top_n]:
            fh.write(f"Variation: {r['variation']}: mean_bias={r['mean_bias']} median={r['median_bias']}\n")

    print("Wrote:")
    print(" -", eval_dir / "scenario_biases.csv")
    print(" -", eval_dir / "variation_biases.csv")
    print(" -", eval_dir / "bias_summary.txt")


if __name__ == '__main__':
    main()
