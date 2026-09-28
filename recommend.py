#!/usr/bin/env python3
"""Train a course ranker and produce an MCU competition submission.

The archive is read directly; no unpacking is needed.  The third-period
answers are used only after a student-wise local validation split.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


WEIGHTS = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.45, 5: 0.35}
CATS = ["module", "institute", "specialty"]
NUMS = [
    "period", "global_rate", "recent_rate", "institute_rate",
    "specialty_rate", "institute_recent_rate", "specialty_recent_rate",
    "global_count", "recent_count", "institute_count", "specialty_count",
    "previous_institute_students", "previous_specialty_students",
    "available_periods", "first_period", "title_length", "description_length",
    "seen_ever", "seen_recently", "best_previous_priority",
    "recent_priority", "previous_occurrences", "history_count",
    "text_similarity_max", "text_similarity_mean", "text_similarity_recent",
    "profile_text_similarity",
]
FEATURES = CATS + NUMS


def read_archive(path: Path):
    with zipfile.ZipFile(path) as archive:
        def jsonl(name):
            return [json.loads(line) for line in archive.read("MCU/" + name).splitlines()]

        train = jsonl("train.jsonl")
        val = jsonl("val.jsonl")
        test = jsonl("test.jsonl")
        modules = jsonl("modules.jsonl")
        solution_text = archive.read("MCU/val_solution.csv").decode("utf-8")
        truth = {}
        for row in csv.DictReader(io.StringIO(solution_text)):
            truth[row["id"]] = {
                item.split(":")[0]: int(item.split(":")[1])
                for item in row["ground_truth"].split()
            }
    return train, val, test, modules, truth


def choice_events(train, val, truth):
    """One event per observed student/period; avoid double counting overlaps."""
    events = {}
    for student in train:
        for entry in student["history"]:
            events[(student["student_id"], entry["period"])] = (
                student["profile"],
                {c["module_id"]: c["priority"] for c in entry["choices"]},
            )
    profiles = {row["id"]: row["profile"] for row in val}
    for sid, answer in truth.items():
        events[(sid, 3)] = profiles[sid], answer
    return events


class FeatureMaker:
    def __init__(self, modules, events):
        self.modules = {m["module_id"]: m for m in modules}
        self.module_ids = sorted(self.modules)
        self.lookup = {m: i for i, m in enumerate(self.module_ids)}
        self.events = events
        self.stats_cache = {}
        text = [
            (self.modules[m]["title"] + " ") * 3
            + self.modules[m]["description"][:1200]
            for m in self.module_ids
        ]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=50000,
            sublinear_tf=True,
        )
        matrix = self.vectorizer.fit_transform(text)
        self.similarity = (matrix @ matrix.T).toarray().astype(np.float32)
        self.module_matrix = matrix
        self.profile_similarity_cache = {}

    def stats(self, target_period):
        if target_period in self.stats_cache:
            return self.stats_cache[target_period]
        n = len(self.module_ids)
        counts = np.zeros(n, dtype=np.float32)
        recent = np.zeros(n, dtype=np.float32)
        inst = defaultdict(lambda: np.zeros(n, dtype=np.float32))
        spec = defaultdict(lambda: np.zeros(n, dtype=np.float32))
        inst_recent = defaultdict(lambda: np.zeros(n, dtype=np.float32))
        spec_recent = defaultdict(lambda: np.zeros(n, dtype=np.float32))
        inst_students = Counter()
        spec_students = Counter()
        observed_students = set()
        for (sid, period), (profile, answer) in self.events.items():
            if period >= target_period:
                continue
            institute = profile.get("institute", "") or "unknown"
            specialty = profile.get("specialty", "") or "unknown"
            observed_students.add(sid)
            inst_students[institute] += 1
            spec_students[specialty] += 1
            for module, priority in answer.items():
                if module not in self.lookup:
                    continue
                j = self.lookup[module]
                weight = WEIGHTS.get(priority, 0.35)
                counts[j] += weight
                inst[institute][j] += weight
                spec[specialty][j] += weight
                if period == target_period - 1:
                    recent[j] += weight
                    inst_recent[institute][j] += weight
                    spec_recent[specialty][j] += weight
        stats = (counts, recent, inst, spec, inst_recent, spec_recent,
                 inst_students, spec_students, len(observed_students))
        self.stats_cache[target_period] = stats
        return stats

    def profile_similarities(self, specialty):
        if specialty not in self.profile_similarity_cache:
            if specialty:
                vector = self.vectorizer.transform([specialty])
                scores = (vector @ self.module_matrix.T).toarray()[0]
            else:
                scores = np.zeros(len(self.module_ids), dtype=np.float32)
            self.profile_similarity_cache[specialty] = scores.astype(np.float32)
        return self.profile_similarity_cache[specialty]

    def frame(self, records, period, *, sample_negatives=False):
        stats = self.stats(period)
        (counts, recent, inst, spec, inst_recent, spec_recent,
         inst_students, spec_students, n_students) = stats
        rows = []
        labels = []
        groups = []
        for record, answer in records:
            sid = record.get("id", record.get("student_id"))
            profile = record["profile"]
            institute = profile.get("institute", "") or "unknown"
            specialty = profile.get("specialty", "") or "unknown"
            available = record.get("available_modules")
            if available is None:
                available = [m for m in self.module_ids
                             if period in self.modules[m]["available_in_periods"]]
            candidates = [m for m in available if m in self.lookup]
            if sample_negatives and answer:
                positive = set(answer)
                negatives = [m for m in candidates if m not in positive]
                popular = sorted(negatives, key=lambda m: -counts[self.lookup[m]])[:25]
                remaining = [m for m in negatives if m not in set(popular)]
                seed = 1000 * period + int(sid.split("_")[-1])
                rng = np.random.default_rng(seed)
                random_part = rng.choice(remaining, size=min(40, len(remaining)),
                                         replace=False).tolist()
                chosen = set(popular + random_part) | positive
                candidates = [m for m in candidates if m in chosen]
            indices = [self.lookup[m] for m in candidates]
            previous = []
            for entry in record["history"]:
                if entry["period"] >= period:
                    continue
                for choice in entry["choices"]:
                    if choice["module_id"] in self.lookup:
                        previous.append((self.lookup[choice["module_id"]],
                                         entry["period"], choice["priority"]))
            hist_indices = [j for j, _, _ in previous]
            similarities = self.similarity[np.ix_(indices, hist_indices)] if previous else None
            hist_weights = np.array([
                WEIGHTS.get(priority, 0.35) * (0.7 ** (period - p - 1))
                for _, p, priority in previous
            ], dtype=np.float32)
            recent_positions = [k for k, (_, p, _) in enumerate(previous) if p == period - 1]
            prof_sims = self.profile_similarities(specialty)
            ic = inst[institute]
            sc = spec[specialty]
            ir = inst_recent[institute]
            sr = spec_recent[specialty]
            for k, (module, j) in enumerate(zip(candidates, indices)):
                metadata = self.modules[module]
                positions = [z for z, (h, _, _) in enumerate(previous) if h == j]
                recent_positions_for_module = [z for z in positions if previous[z][1] == period - 1]
                sims = similarities[k] if similarities is not None else None
                row = {
                    "module": module,
                    "institute": institute,
                    "specialty": specialty,
                    "period": period,
                    "global_rate": counts[j] / max(n_students, 1),
                    "recent_rate": recent[j] / max(n_students, 1),
                    "institute_rate": ic[j] / max(inst_students[institute], 1),
                    "specialty_rate": sc[j] / max(spec_students[specialty], 1),
                    "institute_recent_rate": ir[j] / max(inst_students[institute], 1),
                    "specialty_recent_rate": sr[j] / max(spec_students[specialty], 1),
                    "global_count": counts[j],
                    "recent_count": recent[j],
                    "institute_count": ic[j],
                    "specialty_count": sc[j],
                    "previous_institute_students": inst_students[institute],
                    "previous_specialty_students": spec_students[specialty],
                    "available_periods": len(metadata["available_in_periods"]),
                    "first_period": min(metadata["available_in_periods"]),
                    "title_length": len(metadata["title"]),
                    "description_length": len(metadata["description"]),
                    "seen_ever": int(bool(positions)),
                    "seen_recently": int(bool(recent_positions_for_module)),
                    "best_previous_priority": min((previous[z][2] for z in positions), default=6),
                    "recent_priority": min((previous[z][2] for z in recent_positions_for_module), default=6),
                    "previous_occurrences": len(positions),
                    "history_count": len(previous),
                    "text_similarity_max": float(sims.max()) if sims is not None else 0,
                    "text_similarity_mean": float(np.dot(sims, hist_weights) / hist_weights.sum()) if sims is not None else 0,
                    "text_similarity_recent": float(sims[recent_positions].max()) if recent_positions else 0,
                    "profile_text_similarity": float(prof_sims[j]),
                }
                rows.append(row)
                labels.append(WEIGHTS.get(answer[module], 0.35) if answer and module in answer else 0.0)
                groups.append(sid)
        frame = pd.DataFrame(rows, columns=FEATURES)
        for col in NUMS:
            frame[col] = frame[col].astype(np.float32)
        return frame, np.asarray(labels, dtype=np.float32), groups


def train_model(frame, relevance, *, iterations, eval_set=None):
    binary = (relevance > 0).astype(np.int8)
    # Positive examples carry the official priority weights.
    sample_weight = np.where(binary, relevance, 1.0)
    model = CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        l2_leaf_reg=6,
        random_seed=42,
        thread_count=-1,
        verbose=100,
        allow_writing_files=False,
    )
    kwargs = {}
    if eval_set is not None:
        x_eval, y_eval = eval_set
        kwargs["eval_set"] = (x_eval, (y_eval > 0).astype(np.int8))
        kwargs["early_stopping_rounds"] = 70
    model.fit(frame, binary, cat_features=CATS,
              sample_weight=sample_weight, **kwargs)
    return model


def ranked_predictions(model, frame, groups):
    scores = model.predict_proba(frame)[:, 1]
    results = defaultdict(list)
    for sid, module, score in zip(groups, frame["module"], scores):
        results[sid].append((module, float(score)))
    return {sid: [m for m, _ in sorted(items, key=lambda item: (-item[1], item[0]))[:5]]
            for sid, items in results.items()}


def pndcg(predictions, truth):
    scores = []
    for sid, choices in truth.items():
        dcg = sum(WEIGHTS[choices[module]] / math.log2(position + 2)
                  for position, module in enumerate(predictions[sid]) if module in choices)
        scores.append(min(1.0, dcg))
    return float(np.mean(scores))


def write_submission(path, records, predictions, modules):
    known = {m["module_id"] for m in modules}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "recommendations"])
        for record in records:
            sid = record["id"]
            ranked = predictions[sid]
            assert len(ranked) == len(set(ranked)) == 5
            assert set(ranked) <= set(record["available_modules"]) & known
            writer.writerow([sid, " ".join(ranked)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).parent / "data" / "selection_formula.zip")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "submission.csv")
    parser.add_argument("--iterations", type=int, default=550)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    train, val, test, modules, truth = read_archive(args.archive)
    events = choice_events(train, val, truth)
    maker = FeatureMaker(modules, events)
    p2 = [(row, {c["module_id"]: c["priority"] for entry in row["history"]
                 if entry["period"] == 2 for c in entry["choices"]})
          for row in train if any(entry["period"] == 2 for entry in row["history"])]
    x2, y2, _ = maker.frame(p2, 2, sample_negatives=True)
    p3 = [(row, truth[row["id"]]) for row in val]
    x3, y3, group3 = maker.frame(p3, 3, sample_negatives=True)
    print(f"Training rows: period 2 = {len(x2)}, period 3 = {len(x3)}", flush=True)

    if not args.skip_validation:
        # Split by student. The held-out third-period labels never enter fitting.
        rng = np.random.default_rng(42)
        held_out = set(rng.choice([row["id"] for row in val], size=len(val) // 5,
                                      replace=False))
        train_mask = np.array([sid not in held_out for sid in group3])
        hold_records = [(row, truth[row["id"]]) for row in val if row["id"] in held_out]
        x_hold, y_hold, hold_groups = maker.frame(hold_records, 3)
        x_fit = pd.concat([x2, x3.loc[train_mask]], ignore_index=True)
        y_fit = np.concatenate([y2, y3[train_mask]])
        model = train_model(x_fit, y_fit, iterations=args.iterations,
                            eval_set=(x_hold, y_hold))
        predictions = ranked_predictions(model, x_hold, hold_groups)
        score = pndcg(predictions, {sid: truth[sid] for sid in held_out})
        print(f"Held-out period-3 pNDCG@5: {score:.4f} ({len(held_out)} students)", flush=True)
        iterations = max(100, model.best_iteration_ + 1)
        del model, x_fit, x_hold
    else:
        iterations = args.iterations

    x_fit = pd.concat([x2, x3], ignore_index=True)
    y_fit = np.concatenate([y2, y3])
    model = train_model(x_fit, y_fit, iterations=iterations)
    x_test, _, test_groups = maker.frame([(row, None) for row in test], 4)
    predictions = ranked_predictions(model, x_test, test_groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(args.output, test, predictions, modules)
    print(f"Saved {len(predictions)} recommendations to {args.output}")


if __name__ == "__main__":
    main()
