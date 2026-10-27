# goML take-home submission

| Part | Folder | Contents |
| --- | --- | --- |
| 1 | `placement/` | `analysis.ipynb`, `predict.py`, `requirements.txt` |
| 2 | `class_messages/` | `extract.py`, `NOTES.md`, `requirements.txt` |

## Part 1: placement

```bash
cd placement
pip install -r requirements.txt
python predict.py --input some_students.csv --output predictions.csv
```

`predict.py` loads the pre-trained `model.joblib` (logistic regression without `status_code` or `city_tier`, threshold tuned for F1, CV F1 ≈ 0.56).
It does **not** use `status_code`, which the analysis shows is almost certainly target leakage.
`--use-status-code` switches to `model_with_status_code.joblib` (CV F1 ≈ 0.92) for comparison.
`predict.py` prints warnings (missing columns, unknown departments, out-of-range or non-numeric values) to stderr but always scores every row.
See `analysis.ipynb` for the reasoning (TL;DR at the top).

```bash
python train.py          # rebuild both model files from data/train.csv (deterministic)
python -m pytest tests   # output format, determinism, messy input, leak exclusion
```
