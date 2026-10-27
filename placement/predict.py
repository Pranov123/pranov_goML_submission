import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "model.joblib"
STATUS_MODEL = HERE / "model_with_status_code.joblib"

NUMERIC = ["city_tier", "cgpa", "backlogs", "internship_months", "coding_score", "projects", "workshops", "status_code"]
CATEGORICAL = ["department"]


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

    raw = pd.read_csv(args.input, dtype={"student_id": str})
    if "student_id" not in raw:
        sys.exit("input has no student_id column")

    X = prepare(raw)[bundle["features"]]
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    pred = (proba >= bundle["threshold"]).astype(int)

    pd.DataFrame({"student_id": raw["student_id"], "placed": pred}).to_csv(args.output, index=False)
    print(f"{model_path.name}: wrote {len(pred)} rows to {args.output} "
          f"({pred.sum()} predicted placed, threshold {bundle['threshold']:.2f})")


if __name__ == "__main__":
    main()
