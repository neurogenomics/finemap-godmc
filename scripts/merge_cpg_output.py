#!/usr/bin/env python3
"""
Merge per-CpG SuSiE result files into a single CSV.
"""

import argparse
from pathlib import Path

import pandas as pd


def cpg_from_filename(path: Path) -> str:
    name = path.stem
    if name.endswith("_susie"):
        return name[: -len("_susie")]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-CpG result CSVs into one file"
    )
    parser.add_argument(
        "--input-dir",
        default="data/susie_results",
        help="Directory containing per-CpG CSV files",
    )
    parser.add_argument(
        "--output-path",
        default="data/susie_results_merged.csv",
        help="Path to write the merged CSV",
    )
    parser.add_argument(
        "--pattern",
        default="*_susie.csv",
        help="Glob pattern for input files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{args.pattern}' found in {input_dir}"
        )

    merged_frames = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue

        if "cpg" not in df.columns:
            raise ValueError(f"Missing 'cpg' column in {path}")

        file_cpg = cpg_from_filename(path)
        observed_cpg = df["cpg"].dropna().astype(str).unique()
        if len(observed_cpg) > 1:
            raise ValueError(
                f"Multiple cpg values found in {path}: {observed_cpg.tolist()}"
            )
        if len(observed_cpg) == 1 and observed_cpg[0] != file_cpg:
            raise ValueError(
                f"Filename cpg '{file_cpg}' does not match file contents '{observed_cpg[0]}' in {path}"
            )

        merged_frames.append(df)

    if not merged_frames:
        raise ValueError("All matching files were empty; nothing to merge")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = pd.concat(merged_frames, ignore_index=True)
    merged.to_csv(output_path, index=False)

    print(f"Merged {len(merged_frames)} files into {output_path}")
    print(f"Total rows: {len(merged):,}")


if __name__ == "__main__":
    main()
