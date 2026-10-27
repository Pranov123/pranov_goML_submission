import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "model.joblib"
STATUS_MODEL = HERE / "model_with_status_code.joblib"

NUMERIC = ["city_tier", "cgpa", "backlogs", "internship_months", "coding_score", "projects", "workshops", "status_code"]
CATEGORICAL = ["department"]
DEPARTMENTS = {"CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"}
# Plausible ranges, used only to warn about suspicious input (values are still scored)
RANGES = {"city_tier": (1, 3), "cgpa": (0, 10), "backlogs": (0, None), "internship_months": (0, None),
          "coding_score": (0, 100), "projects": (0, None), "workshops": (0, None), "status_code": (0, 1)}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw rows into the shape the model expects. Used for both training and scoring.

    Every value is allowed to be missing: absent columns are added as NaN, numbers are coerced
    (bad values become NaN), and the pipeline's imputers fill the gaps.
    """
    out = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    out["department"] = df["department"].map(_clean_department) if "department" in df else np.nan
    return out


def _clean_department(v):
    if pd.isna(v):
        return np.nan
    v = str(v).strip().upper()
    return v or np.nan


def check_input(raw: pd.DataFrame, features: list) -> list:
    """Return human-readable warnings about the input. Nothing here stops scoring."""
    warnings = []
    clean = prepare(raw)
    missing_cols = [c for c in features if c not in raw]
    if missing_cols:
        warnings.append(f"columns missing from input, treated as empty: {missing_cols}")
    if raw["student_id"].duplicated().any():
        warnings.append(f"{raw['student_id'].duplicated().sum()} duplicate student_id values")
    for c in features:
        if c not in raw:
            continue
        if c in NUMERIC:
            bad = raw[c].notna() & clean[c].isna()
            if bad.any():
                warnings.append(f"{c}: {bad.sum()} non-numeric values treated as empty")
            lo, hi = RANGES[c]
            out = (lo is not None and (clean[c] < lo)) | (hi is not None and (clean[c] > hi))
            if out.any():
                warnings.append(f"{c}: {out.sum()} values outside expected range {lo}-{hi if hi is not None else ''}")
        share_empty = clean[c].isna().mean()
        if share_empty > 0.5:
            warnings.append(f"{c}: {share_empty:.0%} of values empty")
    if "department" in features and "department" in raw:
        unknown = sorted(set(clean["department"].dropna()) - DEPARTMENTS)
        if unknown:
            warnings.append(f"unknown departments (scored with no department effect): {unknown}")
    return warnings


def predict(raw: pd.DataFrame, bundle: dict) -> pd.Series:
    """0/1 predictions for each row of raw, in the same order."""
    if raw.empty:
        return pd.Series([], dtype=int)
    proba = bundle["pipeline"].predict_proba(prepare(raw)[bundle["features"]])[:, 1]
    return pd.Series((proba >= bundle["threshold"]).astype(int), index=raw.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV of students (same columns as train.csv, without `placed`)")
    ap.add_argument("--output", required=True, help="where to write student_id,placed")
    ap.add_argument("--use-status-code", action="store_true",
                    help="use the model that includes status_code (likely target leakage; see notebook)")
    ap.add_argument("--model", help="path to a saved model bundle (overrides the two options above)")
    args = ap.parse_args()

    model_path = Path(args.model) if args.model else (STATUS_MODEL if args.use_status_code else DEFAULT_MODEL)
    bundle = joblib.load(model_path)
    if bundle.get("sklearn_version") != sklearn.__version__:
        print(f"warning: model was saved with scikit-learn {bundle.get('sklearn_version')}, running {sklearn.__version__}; "
              f"install requirements.txt (or run train.py) if predictions look wrong", file=sys.stderr)

    raw = pd.read_csv(args.input, dtype={"student_id": str})
    if "student_id" not in raw:
        sys.exit("input has no student_id column")

    for w in check_input(raw, bundle["features"]):
        print(f"warning: {w}", file=sys.stderr)

    pred = predict(raw, bundle)
    pd.DataFrame({"student_id": raw["student_id"], "placed": pred}).to_csv(args.output, index=False)
    print(f"{model_path.name}: wrote {len(pred)} rows to {args.output} "
          f"({pred.sum()} predicted placed, threshold {bundle['threshold']:.2f})")


if __name__ == "__main__":
    main()
