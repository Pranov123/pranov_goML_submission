"""Run from the placement/ folder:  python -m pytest tests"""
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predict import DEFAULT_MODEL, STATUS_MODEL, check_input, predict, prepare  # noqa: E402

TRAIN = ROOT / "data" / "train.csv"


@pytest.fixture(scope="module")
def students():
    return pd.read_csv(TRAIN, dtype={"student_id": str}).drop(columns="placed").head(300)


def run_cli(tmp_path, df, *extra):
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    df.to_csv(src, index=False)
    r = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--input", str(src), "--output", str(dst), *extra],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return pd.read_csv(dst, dtype={"student_id": str}), r.stderr


@pytest.mark.parametrize("extra", [[], ["--use-status-code"]])
def test_output_format(tmp_path, students, extra):
    out, _ = run_cli(tmp_path, students, *extra)
    assert list(out.columns) == ["student_id", "placed"]
    assert out.student_id.tolist() == students.student_id.tolist()
    assert set(out.placed) <= {0, 1}


def test_deterministic(tmp_path, students):
    first, _ = run_cli(tmp_path, students)
    second, _ = run_cli(tmp_path, students)
    pd.testing.assert_frame_equal(first, second)


def test_messy_input_still_scores(tmp_path, students):
    messy = students.drop(columns="workshops").astype(object)
    messy.loc[messy.index[0], "department"] = "AI&DS"
    messy.loc[messy.index[1], "department"] = " cse "
    messy.loc[messy.index[2], "cgpa"] = "eight"
    messy.loc[messy.index[3], messy.columns[1:]] = np.nan
    out, stderr = run_cli(tmp_path, messy)
    assert len(out) == len(messy)
    assert "workshops" in stderr and "AI&DS" in stderr and "non-numeric" in stderr


def test_department_normalised():
    clean = prepare(pd.DataFrame({"department": [" cse ", "Mech", "", None]}))
    assert clean.department.iloc[:2].tolist() == ["CSE", "MECH"]
    assert clean.department.iloc[2:].isna().all()


def test_empty_input():
    bundle = joblib.load(DEFAULT_MODEL)
    empty = pd.DataFrame(columns=["student_id", "department", "cgpa"])
    assert predict(empty, bundle).empty


def test_clean_input_has_no_warnings(students):
    for path in (DEFAULT_MODEL, STATUS_MODEL):
        assert check_input(students, joblib.load(path)["features"]) == []


def test_default_model_ignores_status_code(students):
    """The default model must not change its answer when status_code changes (it is excluded as leakage)."""
    bundle = joblib.load(DEFAULT_MODEL)
    assert "status_code" not in bundle["features"]
    a = predict(students.assign(status_code=0), bundle)
    b = predict(students.assign(status_code=1), bundle)
    pd.testing.assert_series_equal(a, b)
