from __future__ import annotations

import json
from pathlib import Path


def infer_first_ab(raw_output: str) -> str | None:
    for char in raw_output:
        if char in {"a", "A"}:
            return "a"
        if char in {"b", "B"}:
            return "b"
    return None


def main() -> None:
    root = Path("output/judgements/phi-4")
    if not root.exists():
        print(f"Folder not found: {root}")
        return

    unknown_files: list[Path] = []
    updated_files: list[Path] = []
    unresolved_files: list[Path] = []

    for json_file in root.rglob("*.json"):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        chosen = str(data.get("chosen_option", "")).strip().lower()
        if chosen == "unknown":
            unknown_files.append(json_file)

            raw_output = str(data.get("raw_output", ""))
            first_ab = infer_first_ab(raw_output)
            options = data.get("options", {})

            if first_ab in {"a", "b"} and isinstance(options, dict):
                replacement = options.get(first_ab)
                if isinstance(replacement, str) and replacement.strip():
                    data["chosen_option"] = replacement
                    try:
                        with json_file.open("w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        updated_files.append(json_file)
                        continue
                    except Exception:
                        pass

            unresolved_files.append(json_file)

    if not unknown_files:
        print("No files with chosen_option = 'unknown' found.")
        return

    for file_path in sorted(unknown_files):
        print(file_path.name)

    print(f"\nFound {len(unknown_files)} file(s) with chosen_option = 'unknown'.")
    print(f"Updated {len(updated_files)} file(s) by parsing raw_output.")
    print(f"Unresolved {len(unresolved_files)} file(s).")


if __name__ == "__main__":
    main()
