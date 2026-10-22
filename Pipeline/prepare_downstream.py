"""Validate and explicitly join D/k estimates to a downstream comparison table.

No input is overwritten. This utility never invents missing comparison columns.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def checked_table(path, required):
    df = pd.read_csv(path, dtype={"id": str})
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError(f"{path}: patient IDs must be nonempty and unique")
    df["id"] = df["id"].str.strip()
    if (df["id"] == "").any() or df["id"].duplicated().any():
        raise ValueError(f"{path}: patient IDs must be nonempty and unique")
    return df


def prepare(estimated, comparison, output):
    est = checked_table(estimated, ["id", "D", "k", "t", "D/k"])
    comp = checked_table(comparison, ["id", "Swanson D/k"])
    for column in ("D", "k", "t", "D/k"):
        values = pd.to_numeric(est[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Estimated {column} must contain finite numeric values")
    d = pd.to_numeric(est["D"]).to_numpy()
    k = pd.to_numeric(est["k"]).to_numpy()
    ratio = pd.to_numeric(est["D/k"]).to_numpy()
    if (d <= 0).any() or (k <= 0).any() or (pd.to_numeric(est["t"]).to_numpy() < 0).any():
        raise ValueError("Estimated D and k must be positive and t nonnegative")
    if not np.allclose(ratio, d / k, rtol=1e-5, atol=0):
        raise ValueError("Estimated D/k must agree with D divided by k")
    comparator = pd.to_numeric(comp["Swanson D/k"], errors="coerce")
    if comparator.notna().sum() == 0:
        raise ValueError("Comparison data contain no numeric Swanson D/k values")
    if (comp["Swanson D/k"].notna() & comparator.isna()).any() or not np.isfinite(comparator.dropna().to_numpy()).all():
        raise ValueError("Swanson D/k must contain numeric finite values or missing values")
    comp["Swanson D/k"] = comparator
    overlap = set(est["id"]) & set(comp["id"])
    if not overlap:
        raise ValueError("No matching patient IDs between estimates and comparison data")
    merged = est.merge(comp[["id", "Swanson D/k"]], on="id", validate="one_to_one", how="left")
    output = Path(output)
    if output.resolve() in (Path(estimated).resolve(), Path(comparison).resolve()):
        raise ValueError("Output must not overwrite either source")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Matched {len(overlap)} of {len(est)} estimated patients to {len(comp)} comparison patients")
    print(f"Saved {output}")
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimated", required=True, help="Pipeline-generated dkDf.csv")
    parser.add_argument("--comparison", required=True, help="Original validated table with id and Swanson D/k")
    parser.add_argument("--output", required=True, help="New CSV output path (never an input path)")
    args = parser.parse_args(argv)
    prepare(args.estimated, args.comparison, args.output)


if __name__ == "__main__":
    main()
