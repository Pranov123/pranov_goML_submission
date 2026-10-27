# goML take-home submission

| Part | Folder | Contents |
| --- | --- | --- |
| 1 | `placement/` | `analysis.ipynb`, `predict.py`, `requirements.txt` (+ `train.py`, `tests/`, saved models) |
| 2 | `class_messages/` | `extract.py`, `NOTES.md`, `requirements.txt` |

---

## Part 1: Will this student get placed?

### Run it

```bash
cd placement
pip install -r requirements.txt
python predict.py --input some_students.csv --output predictions.csv
```

`predict.py` loads the pre-trained `model.joblib` and never fits a model. It handles empty cells, missing columns, unknown departments and junk values, printing warnings to stderr while still scoring every row. Running it twice gives identical output.

```bash
python predict.py ... --use-status-code   # comparison model that uses status_code (see below; not recommended)
python train.py                            # rebuild both model files from data/train.csv (byte-identical, deterministic)
python -m pytest tests                     # 8 tests: format, determinism, messy input, leak exclusion
```

### Result

| | CV F1 (placed = 1) | Notes |
| --- | --- | --- |
| **Default model** (`model.joblib`) | **0.564** | Logistic regression, threshold 0.25 |
| Predict "placed" for everyone | 0.44 | Trivial baseline |
| Model using `status_code` | 0.924 | Collapses when the leak is removed (see below) |

**Expected F1 on a new list:** about 0.56, likely between **0.50 and 0.62** on a list of 500 students (bootstrap).

### Key findings

**1. `status_code` is target leakage, so it is excluded.**
- On its own it reproduces the label: 92% of students with `status_code = 1` were placed, against 2% with 0 (F1 0.94).
- Students with `status_code = 1` have exactly the same feature profile as placed students, so it behaves like a copy of the outcome.
- **Stress test.** On held-back students, the `status_code` model drops from 0.92 to **0.34** when the field is blank and to **0.06** when it is 0. Both are worse than predicting "placed" for everyone. The leak-free model stays at 0.58 in every case.
- **Natural experiment.** On the 113 training students whose `status_code` really is missing, the `status_code` model scores 0.27 and the leak-free model 0.59.
- Conclusion: the field is recorded after the outcome. A truly new list of registered students wouldn't have it.

**2. What drives placement.**
- CGPA and coding score have the strongest effect. Placement rises from about 10% in the lowest tenth of students to 50% or more in the highest.
- Backlogs lower the chance steadily (35% with none, 10% with three).
- Department matters on its own: CSE 38% against CIVIL 14%, even though average CGPA and coding score are the same across departments.
- Internships and projects help a little. City tier and workshops show almost no effect.

**3. Simple beats complex.**
- Logistic regression matched gradient boosting, random forests, splines and interaction terms. All scored 0.55–0.57, within noise.
- Tuning the threshold mattered most: F1 is 0.45 at the default 0.5 and 0.56 at 0.25.
- The threshold is stable across resamples (0.23–0.35), and fixing it at 0.25 costs only 0.003 F1.

**4. Data quality.**
- Missing values: `coding_score` 21%, `backlogs` 6.7%, `cgpa` 4.7%, `status_code` 4.5%.
- They look random: gaps in different columns don't happen together, they're spread evenly across departments, and placement rates are about the same whether a value is present or missing.
- The pipeline fills gaps with the median (most frequent value for department) and adds flags marking which values were missing. It does this for **every** column, because the scoring file can have empty cells anywhere.

**5. Where the model fails.**
- Missed placements look like ordinary unplaced students (average CGPA 7.0, coding score 50). Whatever placed them, such as interviews, referrals or recruiter needs, isn't in the data. This is why AUC stops at about 0.77.
- False alarms look like strong candidates who simply didn't get an offer.
- The predicted probabilities match real placement rates well, so they can be read as actual chances.

**6. Fairness.**
- **`city_tier` is dropped.** It adds nothing (−0.002 F1) and may stand in for family background.
- **`department` is kept, with the trade-off stated openly.** Removing it costs about 0.03 F1. The cost of keeping it is uneven recall: 86% of placed CSE students are caught, but only 37% of placed ECE students. A model without department halves that gap.
- **For targeting support:** of the 20% of students rated least likely to be placed, 93% really were unplaced, against 72% for a random pick. The placement cell should rank students *within* a department so branches with low placement rates aren't written off.

### Files

| File | Purpose |
| --- | --- |
| `analysis.ipynb` | Full analysis: EDA, leak investigation, model comparison, stress test, uncertainty, error analysis, fairness. TL;DR at the top. |
| `predict.py` | Scores a CSV with the saved model and checks the input |
| `train.py` | Rebuilds the saved models; the notebook confirms it reproduces the same threshold and F1 |
| `model.joblib` | Default model (no `status_code`, no `city_tier`) |
| `model_with_status_code.joblib` | Comparison model, used only with `--use-status-code` |
| `tests/test_predict.py` | `pytest` suite |
| `data/train.csv` | Training data, so the notebook can be rerun |

---

## Part 2: Class task list

Work in progress.
