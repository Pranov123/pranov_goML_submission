"""Score a tasks.json against expected answers.

    python eval/score.py --predicted out.json --expected eval/devset_expected.json --inbox eval/devset

Tasks are matched within each message by word overlap, then owner and deadline are compared exactly
(case-insensitive, ignoring a leading "by"). Also counts owners/deadlines that do not appear in the message.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

STOP = {"the", "a", "an", "to", "for", "of", "and", "in", "on", "at", "by", "with", "from", "about", "please"}


def words(s):
    return {w[:5] for w in re.findall(r"[\w@']+", s.lower()) if w not in STOP}


def sim(a, b):
    wa, wb = words(a), words(b)
    return len(wa & wb) / max(len(wa | wb), 1)


def norm(v):
    return None if v is None else re.sub(r"^by\s+", "", re.sub(r"\s+", " ", v).strip().lower())


def bodies(inbox):
    out = {}
    for p in Path(inbox).glob("*.txt"):
        try:
            text = p.read_text(encoding="utf-8").replace("\r\n", "\n")
        except UnicodeDecodeError:
            continue
        header, _, body = text.partition("\n\n")
        mid = next((l.split(":", 1)[1].strip() for l in header.splitlines() if l.startswith("message_id:")), p.stem)
        out[mid] = re.sub(r"\s+", " ", body).lower()
    return out


def score(predicted_tasks, expected_tasks, inbox):
    """Returns (metrics dict, list of problem strings)."""
    pred, exp = defaultdict(list), defaultdict(list)
    for t in predicted_tasks:
        pred[t["message_id"]].append(t)
    for t in expected_tasks:
        exp[t["message_id"]].append(t)
    texts = bodies(inbox)

    tp = fp = fn = owner_ok = due_ok = invented = 0
    problems = []
    for mid in sorted(set(texts) | set(pred) | set(exp)):
        p_left, e_left = list(pred[mid]), list(exp[mid])
        pairs = sorted(((sim(p["task"], e["task"]), i, j) for i, p in enumerate(p_left) for j, e in enumerate(e_left)), reverse=True)
        used_p, used_e = set(), set()
        for s, i, j in pairs:
            if s < 0.3 or i in used_p or j in used_e:
                continue
            used_p.add(i); used_e.add(j); tp += 1
            p, e = p_left[i], e_left[j]
            if norm(p["owner"]) == norm(e["owner"]):
                owner_ok += 1
            else:
                problems.append(f"{mid}: owner {p['owner']!r}, expected {e['owner']!r}  [{e['task']}]")
            if norm(p["due"]) == norm(e["due"]):
                due_ok += 1
            else:
                problems.append(f"{mid}: due {p['due']!r}, expected {e['due']!r}  [{e['task']}]")
        for i, p in enumerate(p_left):
            body = texts.get(mid, "")
            for field in ("owner", "due"):
                if p[field] and norm(p[field]) not in body:
                    invented += 1
                    problems.append(f"{mid}: {field} {p[field]!r} is NOT in the message")
            if i not in used_p:
                fp += 1
                problems.append(f"{mid}: extra task {p['task']!r}")
        for j, e in enumerate(e_left):
            if j not in used_e:
                fn += 1
                problems.append(f"{mid}: missed task {e['task']!r}")

    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    result = {"task_precision": round(prec, 3), "task_recall": round(rec, 3),
              "task_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
              "owner_accuracy": round(owner_ok / max(tp, 1), 3), "due_accuracy": round(due_ok / max(tp, 1), 3),
              "invented_owner_or_due": invented, "matched": tp, "extra": fp, "missed": fn}
    return result, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicted", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--inbox", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))["tasks"]
    result, problems = score(load(args.predicted), load(args.expected), args.inbox)
    print(json.dumps(result, indent=2))
    if not args.quiet:
        print("\n".join(problems) or "no problems")


if __name__ == "__main__":
    main()
