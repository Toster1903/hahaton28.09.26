#!/usr/bin/env python3
"""Train a course ranker and produce an MCU competition submission.

The archive is read directly; no unpacking is needed.  The third-period
answers are used only after a student-wise local validation split.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


WEIGHTS = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.45, 5: 0.35}
CATS = ["module", "institute", "specialty"]
NUMS = [
    "period", "global_rate", "recent_rate", "institute_rate",
    "specialty_rate", "institute_recent_rate", "specialty_recent_rate",
    "global_count", "recent_count", "institute_count", "specialty_count",
    "previous_institute_students", "previous_specialty_students",
    "available_periods", "first_period", "title_length", "description_length",
    "favorites_count",
    "seen_ever", "seen_recently", "best_previous_priority",
    "recent_priority", "previous_occurrences", "history_count",
    "text_similarity_max", "text_similarity_mean", "text_similarity_recent",
    "profile_text_similarity", "cochoice_max", "cochoice_mean",
    "cochoice_recent", "transition_max", "transition_mean",
    "cochoice_cosine_max", "cochoice_cosine_mean",
    "transition_cosine_max", "transition_cosine_mean",
    "title_similarity_max", "title_similarity_mean",
    "description_similarity_max", "description_similarity_mean",
    "history_unique_count", "history_repeat_count", "history_topic_coherence",
    "content_global_rate", "content_recent_rate", "content_institute_rate",
    "content_specialty_rate", "smoothed_global_rate", "smoothed_institute_rate",
    "smoothed_specialty_rate", "course_age",
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
        self.favorites_count = {}
        for m in modules:
            match = re.search(r"Количество обучающихся[^.]{0,120}?добавили данный модуль в избранное\s*(\d+)",
                              m["description"], flags=re.IGNORECASE)
            self.favorites_count[m["module_id"]] = int(match.group(1)) if match else -1
        self.module_ids = sorted(self.modules)
        self.lookup = {m: i for i, m in enumerate(self.module_ids)}
        self.events = events
        self.stats_cache = {}
        self.association_cache = {}
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
        title_vectorizer = TfidfVectorizer(
            analyzer="word", token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 2), sublinear_tf=True,
        )
        title_matrix = title_vectorizer.fit_transform(
            [self.modules[m]["title"] for m in self.module_ids]
        )
        self.title_similarity = (title_matrix @ title_matrix.T).toarray().astype(np.float32)
        desc_vectorizer = TfidfVectorizer(
            analyzer="word", token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 2), max_features=40000, sublinear_tf=True,
        )
        desc_matrix = desc_vectorizer.fit_transform(
            [self.modules[m]["description"][:1800] for m in self.module_ids]
        )
        self.description_similarity = (desc_matrix @ desc_matrix.T).toarray().astype(np.float32)
        self.profile_similarity_cache = {}
        self.content_weight_cache = {}
        self.content_prior_cache = {}

    def content_priors(self, period, institute, specialty):
        key = (period, institute, specialty)
        if key in self.content_prior_cache:
            return self.content_prior_cache[key]
        (counts, recent, inst, spec, _, _, _, _, _, exposure,
         inst_exposure, spec_exposure, recent_students, _, _) = self.stats(period)
        rate = counts / np.maximum(exposure, 1)
        if period not in self.content_weight_cache:
            similarity = (self.similarity + self.title_similarity + self.description_similarity) / 3
            similarity = similarity.copy()
            np.fill_diagonal(similarity, 0)
            similarity[:, exposure == 0] = 0
            neighbors = np.argsort(-similarity, axis=1)[:, :10]
            weights = np.zeros_like(similarity)
            np.put_along_axis(weights, neighbors,
                              np.take_along_axis(similarity, neighbors, axis=1) ** 2, axis=1)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            self.content_weight_cache[period] = weights
        weights = self.content_weight_cache[period]
        content_global = weights @ rate
        content_recent = weights @ (recent / max(recent_students, 1))
        global_prior = (counts + 30 * content_global) / (exposure + 30)
        inst_rate = (inst[institute] + 20 * global_prior) / (inst_exposure[institute] + 20)
        spec_rate = (spec[specialty] + 20 * global_prior) / (spec_exposure[specialty] + 20)
        result = (content_global, content_recent, weights @ inst_rate, weights @ spec_rate,
                  global_prior, inst_rate, spec_rate)
        self.content_prior_cache[key] = result
        return result

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
        exposure = np.zeros(n, dtype=np.float32)
        inst_exposure = defaultdict(lambda: np.zeros(n, dtype=np.float32))
        spec_exposure = defaultdict(lambda: np.zeros(n, dtype=np.float32))
        recent_students = 0
        inst_recent_students = Counter()
        spec_recent_students = Counter()
        period_masks = {p: np.array([p in self.modules[m]["available_in_periods"]
                                    for m in self.module_ids]) for p in range(1, target_period)}
        observed_students = set()
        for (sid, period), (profile, answer) in self.events.items():
            if period >= target_period:
                continue
            institute = profile.get("institute", "") or "unknown"
            specialty = profile.get("specialty", "") or "unknown"
            observed_students.add(sid)
            inst_students[institute] += 1
            spec_students[specialty] += 1
            exposure[period_masks[period]] += 1
            inst_exposure[institute][period_masks[period]] += 1
            spec_exposure[specialty][period_masks[period]] += 1
            if period == target_period - 1:
                recent_students += 1
                inst_recent_students[institute] += 1
                spec_recent_students[specialty] += 1
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
                 inst_students, spec_students, len(observed_students), exposure,
                 inst_exposure, spec_exposure, recent_students,
                 inst_recent_students, spec_recent_students)
        self.stats_cache[target_period] = stats
        return stats

    def associations(self, target_period):
        if target_period in self.association_cache:
            return self.association_cache[target_period]
        n = len(self.module_ids)
        co = np.zeros((n, n), dtype=np.float32)
        transitions = np.zeros((n, n), dtype=np.float32)
        occurrence = np.zeros(n, dtype=np.float32)
        transition_occurrence = np.zeros(n, dtype=np.float32)
        destination_occurrence = np.zeros(n, dtype=np.float32)
        by_student = defaultdict(dict)
        for (sid, period), (_, answer) in self.events.items():
            if period < target_period:
                by_student[sid][period] = [self.lookup[m] for m in answer if m in self.lookup]
        for periods in by_student.values():
            for period, chosen in periods.items():
                for source in chosen:
                    occurrence[source] += 1
                    for target in chosen:
                        if target != source:
                            co[source, target] += 1
                if period + 1 in periods:
                    for target in periods[period + 1]:
                        destination_occurrence[target] += 1
                    for source in chosen:
                        transition_occurrence[source] += 1
                        for target in periods[period + 1]:
                            transitions[source, target] += 1
        co_cosine = co / np.maximum(np.sqrt(occurrence[:, None] * occurrence[None, :]), 1)
        transition_cosine = transitions / np.maximum(
            np.sqrt(transition_occurrence[:, None] * destination_occurrence[None, :]), 1
        )
        co /= np.maximum(occurrence[:, None], 1)
        transitions /= np.maximum(transition_occurrence[:, None], 1)
        self.association_cache[target_period] = co, transitions, co_cosine, transition_cosine
        return self.association_cache[target_period]

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
        cochoice, transitions, co_cosine, trans_cosine = self.associations(period)
        (counts, recent, inst, spec, inst_recent, spec_recent,
         inst_students, spec_students, n_students, exposure,
         inst_exposure, spec_exposure, recent_students,
         inst_recent_students, spec_recent_students) = stats
        rows = []
        labels = []
        groups = []
        for record, answer in records:
            sid = record.get("id", record.get("student_id"))
            profile = record["profile"]
            institute = profile.get("institute", "") or "unknown"
            specialty = profile.get("specialty", "") or "unknown"
            available = record.get("available_modules")
            content_priors = self.content_priors(period, institute, specialty)
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
            title_sims = self.title_similarity[np.ix_(indices, hist_indices)] if previous else None
            desc_sims = self.description_similarity[np.ix_(indices, hist_indices)] if previous else None
            co_scores = cochoice[np.ix_(hist_indices, indices)].T if previous else None
            transition_scores = transitions[np.ix_(hist_indices, indices)].T if previous else None
            co_cos_scores = co_cosine[np.ix_(hist_indices, indices)].T if previous else None
            trans_cos_scores = trans_cosine[np.ix_(hist_indices, indices)].T if previous else None
            hist_weights = np.array([
                WEIGHTS.get(priority, 0.35) * (0.7 ** (period - p - 1))
                for _, p, priority in previous
            ], dtype=np.float32)
            recent_positions = [k for k, (_, p, _) in enumerate(previous) if p == period - 1]
            if len(hist_indices) > 1:
                history_similarities = self.similarity[np.ix_(hist_indices, hist_indices)]
                history_coherence = float((history_similarities.sum() - len(hist_indices)) /
                                          (len(hist_indices) * (len(hist_indices) - 1)))
            else:
                history_coherence = 0.0
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
                co_sims = co_scores[k] if co_scores is not None else None
                tr_sims = transition_scores[k] if transition_scores is not None else None
                co_cos_sims = co_cos_scores[k] if co_cos_scores is not None else None
                tr_cos_sims = trans_cos_scores[k] if trans_cos_scores is not None else None
                title_row = title_sims[k] if title_sims is not None else None
                desc_row = desc_sims[k] if desc_sims is not None else None
                row = {
                    "module": module,
                    "institute": institute,
                    "specialty": specialty,
                    "period": period,
                    "global_rate": counts[j] / max(exposure[j], 1),
                    "recent_rate": recent[j] / max(recent_students, 1),
                    "institute_rate": ic[j] / max(inst_exposure[institute][j], 1),
                    "specialty_rate": sc[j] / max(spec_exposure[specialty][j], 1),
                    "institute_recent_rate": ir[j] / max(inst_recent_students[institute], 1),
                    "specialty_recent_rate": sr[j] / max(spec_recent_students[specialty], 1),
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
                    "favorites_count": self.favorites_count[module],
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
                    "cochoice_max": float(co_sims.max()) if co_sims is not None else 0,
                    "cochoice_mean": float(np.dot(co_sims, hist_weights) / hist_weights.sum()) if co_sims is not None else 0,
                    "cochoice_recent": float(co_sims[recent_positions].max()) if recent_positions else 0,
                    "transition_max": float(tr_sims.max()) if tr_sims is not None else 0,
                    "transition_mean": float(np.dot(tr_sims, hist_weights) / hist_weights.sum()) if tr_sims is not None else 0,
                    "cochoice_cosine_max": float(co_cos_sims.max()) if co_cos_sims is not None else 0,
                    "cochoice_cosine_mean": float(np.dot(co_cos_sims, hist_weights) / hist_weights.sum()) if co_cos_sims is not None else 0,
                    "transition_cosine_max": float(tr_cos_sims.max()) if tr_cos_sims is not None else 0,
                    "transition_cosine_mean": float(np.dot(tr_cos_sims, hist_weights) / hist_weights.sum()) if tr_cos_sims is not None else 0,
                    "title_similarity_max": float(title_row.max()) if title_row is not None else 0,
                    "title_similarity_mean": float(np.dot(title_row, hist_weights) / hist_weights.sum()) if title_row is not None else 0,
                    "description_similarity_max": float(desc_row.max()) if desc_row is not None else 0,
                    "description_similarity_mean": float(np.dot(desc_row, hist_weights) / hist_weights.sum()) if desc_row is not None else 0,
                    "history_unique_count": len(set(hist_indices)),
                    "history_repeat_count": len(hist_indices) - len(set(hist_indices)),
                    "history_topic_coherence": history_coherence,
                    "content_global_rate": content_priors[0][j],
                    "content_recent_rate": content_priors[1][j],
                    "content_institute_rate": content_priors[2][j],
                    "content_specialty_rate": content_priors[3][j],
                    "smoothed_global_rate": content_priors[4][j],
                    "smoothed_institute_rate": content_priors[5][j],
                    "smoothed_specialty_rate": content_priors[6][j],
                    "course_age": period - min(metadata["available_in_periods"]),
                }
                rows.append(row)
                labels.append(WEIGHTS.get(answer[module], 0.35) if answer and module in answer else 0.0)
                groups.append(sid)
        frame = pd.DataFrame(rows, columns=FEATURES)
        for col in NUMS:
            frame[col] = frame[col].astype(np.float32)
        return frame, np.asarray(labels, dtype=np.float32), groups


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
