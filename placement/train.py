import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predict import DEFAULT_MODEL, STATUS_MODEL, prepare

# Default model: no status_code (target leakage) and no city_tier (no signal, possible proxy)
DEFAULT_NUMERIC = ["cgpa", "backlogs", "internship_months", "coding_score", "projects", "workshops"]
STATUS_NUMERIC = ["city_tier", "cgpa", "backlogs", "internship_months", "coding_score", "projects", "workshops", "status_code"]
THRESHOLDS = np.round(np.arange(0.10, 0.80, 0.01), 2)


def build_model(numeric):
    return make_pipeline(
        make_column_transformer(
            (make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
            (make_pipeline(SimpleImputer(strategy="most_frequent"),
                           OneHotEncoder(handle_unknown="ignore", sparse_output=False)), ["department"]),
        ),
        LogisticRegression(C=1.0, max_iter=5000),
    )


def cv_threshold(model, X, y, n_repeats=3, seed=42):
    """Threshold that maximises F1 averaged over repeated 5-fold out-of-fold predictions."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    curves = np.zeros((n_repeats, len(THRESHOLDS)))
    oof = np.zeros((n_repeats, len(y)))
    for i, (tr, te) in enumerate(cv.split(X, y)):
        oof[i // 5, te] = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    for r, p in enumerate(oof):
        pred = p[None, :] >= THRESHOLDS[:, None]
        tp = (pred & (y == 1)).sum(1); fp = (pred & (y == 0)).sum(1); fn = (~pred & (y == 1)).sum(1)
        curves[r] = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    curve = curves.mean(0)
    return float(THRESHOLDS[curve.argmax()]), float(curve.max())


def train_bundle(numeric, X, y):
    model = build_model(numeric)
    threshold, cv_f1 = cv_threshold(model, X, y)
    return {"pipeline": clone(model).fit(X, y), "features": numeric + ["department"], "threshold": threshold,
            "cv_f1": cv_f1, "sklearn_version": sklearn.__version__}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data" / "train.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.data, dtype={"student_id": str})
    X, y = prepare(df), df["placed"].values
    for path, numeric in [(DEFAULT_MODEL, DEFAULT_NUMERIC), (STATUS_MODEL, STATUS_NUMERIC)]:
        bundle = train_bundle(numeric, X, y)
        joblib.dump(bundle, path)
        print(f"{path.name}: threshold={bundle['threshold']:.2f}  CV F1={bundle['cv_f1']:.3f}")


if __name__ == "__main__":
    main()
