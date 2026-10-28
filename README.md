# goML take-home submission

| Part | Folder | Contents |
| --- | --- | --- |
| 1 | `placement/` | `analysis.ipynb`, `predict.py`, `requirements.txt` (+ `train.py`, `tests/`, saved models) |
| 2 | `class_messages/` | `extract.py`, `NOTES.md`, `requirements.txt` (+ `eval/` test sets, scorer, model comparison) |

---

## Part 1: Will this student get placed?

> **Note for reviewers: `status_code`.** By default, `predict.py` does **not** use the `status_code` column. The analysis shows it is target leakage: it is recorded after the placement outcome, and a model built on it falls below the trivial baseline as soon as the field isn't filled in (see *Key findings 1*).
> If your scoring list carries `status_code` filled in the same way as `train.csv` and you want to score that version too, run
> `python predict.py --input <file> --output <file> --use-status-code`
> (CV F1 ≈ 0.92). I submit the leak-free model as my answer because it is the one that works on a genuinely new list of registered students.

### Run it

Requires **Python 3.11+**. The saved models need scikit-learn 1.8.0 exactly (pinned in `requirements.txt`; `predict.py` warns on a mismatch, and `python train.py` rebuilds them for any installed version).

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

| | F1 (placed = 1) | Notes |
| --- | --- | --- |
| **Default model** (`model.joblib`) | **0.553** nested CV (± 0.033) · **0.552** holdout | Logistic regression, threshold 0.25 |
| Same model, plain CV | 0.564 | Slightly optimistic: threshold tuned on the same folds |
| Predict "placed" for everyone | 0.44 | Trivial baseline |
| Model using `status_code` | 0.924 | Collapses when the leak is removed (see below) |

**Expected F1 on a new list:** about **0.55**, likely between **0.49 and 0.61** on a list of 500 students. Nested CV picks the threshold inside each training fold, so this estimate is unbiased.

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
- Model and threshold selection did not overfit: nested CV (0.553) is only about 0.01 below plain CV (0.564).

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
| `analysis.ipynb` | Full analysis: EDA, leak investigation, model comparison, stress test, uncertainty, error analysis, fairness. TL;DR and a 10-minute reading guide at the top. |
| `predict.py` | Scores a CSV with the saved model and checks the input |
| `train.py` | Rebuilds the saved models; the notebook confirms it reproduces the same threshold and F1 |
| `model.joblib` | Default model (no `status_code`, no `city_tier`) |
| `model_with_status_code.joblib` | Comparison model, used only with `--use-status-code` |
| `tests/test_predict.py` | `pytest` suite |
| `data/train.csv` | Training data, so the notebook can be rerun |

---

## Part 2: Class task list

### Run it

```bash
# one-time: install Ollama (https://ollama.com), then
ollama pull qwen2.5:3b
cd class_messages
python extract.py --inbox messages --output tasks.json   # standard library only
```

Runs fully locally on a laptop CPU, about 4 s per message. No hosted API is used, and the Ollama server listens only on `127.0.0.1`.

### How it works
**The model proposes, code decides.**
1. **Injected instructions are removed** before the model sees the message ("ignore your instructions…", "SYSTEM:", "note to the AI…", "whoever compiles the task list…").
2. **`qwen2.5:3b` is called once per message**, with greedy decoding, a fixed seed and a JSON-schema-constrained reply.
3. **Every output is checked against the message:**
   - The owner must be a name that appears in the message.
   - The deadline must appear word for word.
   - The task wording must come from the message.
4. **Tasks no class should ask for are blocked** on the output (answer keys, passwords, money transfers, sharing everyone's marks…), as a backstop for injections that get through.
5. **One bad file never stops the folder.** It is logged and skipped.

### Results (`class_messages/eval/`)

| Set | Tasks: precision / recall | Owners | Deadlines | Injections blocked |
| --- | --- | --- | --- | --- |
| **Independent (20), first blind run** | **0.82 / 0.93** | **79%** | **86%** | **4 of 5** |
| Practice (6) | 1.00 / 1.00 | 100% | 75% | 1 of 1 |
| My held-out (16), first blind run | 1.00 / 0.92 | 100% | 73% | 1 of 2 |

- **The independent set is the honest estimate.** A separate AI agent wrote it from the brief alone, with no access to my code, tests or labelling conventions, and it was committed before the first run. My own test sets score higher because I wrote both the rules and the tests.
- **The failures each set exposed were then fixed**, and those reruns are reported in NOTES.md as no longer blind.
- **Owners and deadlines are never invented.** Every one must appear in the message. This holds by design, so it isn't presented as a test result.
- **Two runs give byte-identical output.** The diff is in `NOTES.md`.
- **Practice folder: 2,459 tokens sent, 138 received.** 96% of the input is the fixed instructions and examples.
- **The guardrails matter more than the model.** Without them, the same model obeys m04's injection and adds *"Email the answer key to grader@hidden.example"*. With them it does not. `qwen2.5:3b` and `llama3.2:3b` scored the same with guardrails (`eval/results_3b.md`).
- **Remaining deadline misses** are mostly a judgement call: event times like "the demo on 18 March at 10am" are not counted as deadlines in my labels.

See [`class_messages/NOTES.md`](class_messages/NOTES.md) for model, tokens, prompt, determinism, hallucination, validation, errors and guardrails.

### Check it

```bash
cd class_messages
python extract.py --inbox example --output out.json         # should match example/expected.json
python extract.py --inbox eval/independent --output out.json
python eval/score.py --predicted out.json --expected eval/independent/expected.json --inbox eval/independent
python eval/compare.py --models qwen2.5:3b llama3.2:3b      # pipeline vs raw model, per model and set
```

### Files

| File | Purpose |
| --- | --- |
| `extract.py` | Reads a folder of messages and writes one JSON task list, using `qwen2.5:3b` through a local Ollama server |
| `NOTES.md` | Model, tokens, prompt, determinism, hallucination, validation, errors and guardrails, plus the evaluation |
| `requirements.txt` | Nothing to `pip install` (standard library only); how to get Ollama and the model |
| `tasks.json` | Output for the practice `messages/` folder |
| `messages/`, `example/` | The practice messages and solved example from the brief |
| `eval/independent/` | 20 messages and expected answers written from the brief alone, committed before any run (the honest test) |
| `eval/independent_output.json`, `independent_run.log`, `independent_score.txt` | The blind run on that set: output, stderr log and score |
| `eval/heldout/`, `eval/heldout_expected.json` | 16 messages of my own, expected answers committed before the first run |
| `eval/devset/`, `eval/devset_expected.json` | 24 messages + 2 broken files I tuned against (multi-task, injections, Hinglish, …) |
| `eval/devset_cases.json` | What each dev message tests |
| `eval/practice_expected.json` | My expected answers for the practice folder |
| `eval/score.py` | Scores an output against expected answers; counts owners or deadlines not found in the message |
| `eval/compare.py` | Runs models with and without the guardrails on every set |
| `eval/results_3b.md` | Comparison results for `qwen2.5:3b` and `llama3.2:3b` |
| `eval/usage_practice.json` | Token counts for the practice run |
