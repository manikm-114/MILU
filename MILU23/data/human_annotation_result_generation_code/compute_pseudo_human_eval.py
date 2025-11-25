import json
import csv
from collections import defaultdict, Counter
from pathlib import Path
import re

# ============ CONFIG ============
GROUND_TRUTH_PATH = Path("ground_truth_human_reference.jsonl")
MODEL_OUTPUTS_PATH = Path("lecture_outputs.jsonl")  # or lecture_outputs_robust.jsonl
OUT_SLIDE_CSV = Path("human_ref_eval_by_slide.csv")
OUT_LECTURE_CSV = Path("human_ref_eval_by_lecture.csv")
OUT_GLOBAL_CSV = Path("human_ref_eval_global_summary.csv")
# ================================


# ---------- Helpers for normalization ----------

def normalize_text(s):
    if s is None:
        return ""
    # basic normalization: lowercase + collapse spaces + strip punctuation at ends
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_concept_item(c):
    """
    Handle several formats:
      - string
      - {"term": "...", ...}
      - {"concept": "...", ...}
      - generic dict -> join string values
    """
    if isinstance(c, str):
        return normalize_text(c)
    if isinstance(c, dict):
        # common keys first
        for k in ["term", "concept", "name", "label"]:
            if k in c and isinstance(c[k], str):
                return normalize_text(c[k])
        # fallback: concatenate all string values
        parts = [str(v) for v in c.values() if isinstance(v, str)]
        if parts:
            return normalize_text(" ".join(parts))
    # last resort
    return normalize_text(str(c))


def extract_concept_set(obj):
    """
    obj: dict expected to have 'concepts' key (list)
    returns: set of normalized concept strings
    """
    concepts = obj.get("concepts", [])
    return {normalize_concept_item(c) for c in concepts if normalize_concept_item(c)}


def normalize_triple_item(t):
    """
    Expect something like:
      {"head": "...", "relation": "...", "tail": "..."}
    but be robust to variations.
    """
    if not isinstance(t, dict):
        # if it's something else, try to stringify
        return (normalize_text(str(t)), "", "")

    # Try to find head/relation/tail with flexible key names
    def first_match(d, candidates, default=""):
        for key in candidates:
            if key in d and isinstance(d[key], str):
                return normalize_text(d[key])
        return default

    head = first_match(t, ["head", "subject", "source"])
    rel = first_match(t, ["relation", "predicate", "rel"])
    tail = first_match(t, ["tail", "object", "target"])
    return (head, rel, tail)


def extract_triple_set(obj):
    """
    obj: dict expected to have 'triples' key (list)
    returns: set of (head, relation, tail) tuples
    """
    triples = obj.get("triples", [])
    return {normalize_triple_item(tr) for tr in triples if any(normalize_triple_item(tr))}


def jaccard(a, b):
    if not a and not b:
        return 1.0  # both empty -> perfect match (you can change to 0.0 if you prefer)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def triple_precision_recall_f1(ref_set, pred_set):
    """
    Treat triples as exact-matching tuples.
    ref_set: set of gold (pseudo-human) triples
    pred_set: set of model triples
    """
    if not ref_set and not pred_set:
        return 1.0, 1.0, 1.0

    tp = len(ref_set & pred_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


# ---------- 1. Load ground-truth (pseudo-human) ----------

print(f"Loading ground truth from {GROUND_TRUTH_PATH} ...")
gt_by_slide = {}  # key: slide_number (int)  -> {"concepts": set, "triples": set}

with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        # You showed: {"slide_number": 1, "content": "...", "concepts": [...], "triples": [...], "raw_output": "..."}
        slide_num = obj.get("slide_number")
        if slide_num is None:
            continue
        concepts_set = extract_concept_set(obj)
        triples_set = extract_triple_set(obj)
        gt_by_slide[int(slide_num)] = {
            "concepts": concepts_set,
            "triples": triples_set,
        }

print(f"Loaded pseudo-human reference for {len(gt_by_slide)} slides.")


# ---------- 2. Load model outputs ----------

print(f"Loading model outputs from {MODEL_OUTPUTS_PATH} ...")

# slide key: integer slide number extracted from "Slide44" -> 44
def slide_id_to_int(slide_id):
    # slide_id like "Slide44" or "Slide_44" etc.
    if not slide_id:
        return None
    nums = re.findall(r"\d+", slide_id)
    if not nums:
        return None
    return int(nums[-1])

# Store per slide per model
# key: (lecture, slide_number, model, type) but lecture can be None
# However for GT we only have slide_number -> we will key by slide_number
records = []

with MODEL_OUTPUTS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)

        lecture = obj.get("lecture")  # e.g., "Lecture 1"
        meta = obj.get("metadata", {})
        lecture_num = meta.get("lecture_number", None)

        out_type = obj.get("type")  # "concepts" or "triples"
        data = obj.get("data", {})
        model_name = data.get("model")  # e.g., "OpenGVLab/InternVL3-14B"
        slide_id = data.get("slide_id")  # e.g., "Slide44"
        slide_num = slide_id_to_int(slide_id)

        if slide_num is None or model_name is None:
            continue

        parsed = data.get("parsed", {})

        # Handle missing or None parsed dict
        if parsed is None or parsed == "" or parsed == []:
            parsed = {}

        # Extract model-side concepts or triples
        if out_type == "concepts":
            try:
                model_concepts = extract_concept_set(parsed)
            except Exception:
                model_concepts = set()
            model_triples = None

        elif out_type == "triples":
            try:
                model_triples = extract_triple_set(parsed)
            except Exception:
                model_triples = set()
            model_concepts = None

        else:
            continue


        records.append({
            "lecture": lecture,
            "lecture_num": lecture_num,
            "slide_num": slide_num,
            "model": model_name,
            "type": out_type,
            "concepts": model_concepts,
            "triples": model_triples,
        })

print(f"Loaded {len(records)} model records.")


# ---------- 3. Compute per-slide metrics vs pseudo-human ----------

slide_metrics = []  # will be written to CSV

for rec in records:
    slide_num = rec["slide_num"]
    model = rec["model"]
    lecture = rec["lecture"]
    lecture_num = rec["lecture_num"]
    out_type = rec["type"]

    gt = gt_by_slide.get(slide_num)
    if gt is None:
        # no pseudo-human reference for this slide
        continue

    row = {
        "lecture": lecture,
        "lecture_num": lecture_num,
        "slide_number": slide_num,
        "model": model,
        "type": out_type,
    }

    if out_type == "concepts":
        gt_concepts = gt["concepts"]
        model_concepts = rec["concepts"] or set()
        j = jaccard(gt_concepts, model_concepts)
        row["concept_jaccard"] = j
        row["triple_precision"] = ""
        row["triple_recall"] = ""
        row["triple_f1"] = ""
    elif out_type == "triples":
        gt_triples = gt["triples"]
        model_triples = rec["triples"] or set()
        prec, rec_, f1 = triple_precision_recall_f1(gt_triples, model_triples)
        row["concept_jaccard"] = ""
        row["triple_precision"] = prec
        row["triple_recall"] = rec_
        row["triple_f1"] = f1
    else:
        continue

    slide_metrics.append(row)

print(f"Computed slide-level metrics for {len(slide_metrics)} entries.")


# ---------- 4. Aggregate per lecture & global ----------

# Per lecture per model:
lecture_agg = defaultdict(lambda: {"concept_jaccard_sum": 0.0,
                                   "concept_jaccard_count": 0,
                                   "triple_f1_sum": 0.0,
                                   "triple_f1_count": 0})

# Global per model:
global_agg = defaultdict(lambda: {"concept_jaccard_sum": 0.0,
                                  "concept_jaccard_count": 0,
                                  "triple_f1_sum": 0.0,
                                  "triple_f1_count": 0})

for row in slide_metrics:
    model = row["model"]
    lecture = row["lecture"]
    lecture_num = row["lecture_num"]
    out_type = row["type"]

    key_lecture = (lecture, lecture_num, model)

    if out_type == "concepts" and row["concept_jaccard"] != "":
        j = float(row["concept_jaccard"])
        lecture_agg[key_lecture]["concept_jaccard_sum"] += j
        lecture_agg[key_lecture]["concept_jaccard_count"] += 1
        global_agg[model]["concept_jaccard_sum"] += j
        global_agg[model]["concept_jaccard_count"] += 1

    if out_type == "triples" and row["triple_f1"] != "":
        f1 = float(row["triple_f1"])
        lecture_agg[key_lecture]["triple_f1_sum"] += f1
        lecture_agg[key_lecture]["triple_f1_count"] += 1
        global_agg[model]["triple_f1_sum"] += f1
        global_agg[model]["triple_f1_count"] += 1


# ---------- 5. Write CSVs ----------

# 5.1 Slide-level CSV
print(f"Writing slide-level metrics to {OUT_SLIDE_CSV} ...")
with OUT_SLIDE_CSV.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "lecture", "lecture_num", "slide_number",
        "model", "type",
        "concept_jaccard",
        "triple_precision", "triple_recall", "triple_f1",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in slide_metrics:
        writer.writerow(row)


# 5.2 Lecture-level CSV
print(f"Writing lecture-level summary to {OUT_LECTURE_CSV} ...")
with OUT_LECTURE_CSV.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "lecture", "lecture_num", "model",
        "avg_concept_jaccard",
        "avg_triple_f1",
        "n_concept_slides",
        "n_triple_slides",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for (lecture, lecture_num, model), stats in lecture_agg.items():
        cj_count = stats["concept_jaccard_count"]
        tf_count = stats["triple_f1_count"]
        avg_cj = (stats["concept_jaccard_sum"] / cj_count) if cj_count > 0 else ""
        avg_tf1 = (stats["triple_f1_sum"] / tf_count) if tf_count > 0 else ""
        writer.writerow({
            "lecture": lecture,
            "lecture_num": lecture_num,
            "model": model,
            "avg_concept_jaccard": avg_cj,
            "avg_triple_f1": avg_tf1,
            "n_concept_slides": cj_count,
            "n_triple_slides": tf_count,
        })


# 5.3 Global summary CSV
print(f"Writing global summary to {OUT_GLOBAL_CSV} ...")
with OUT_GLOBAL_CSV.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "model",
        "global_avg_concept_jaccard",
        "global_avg_triple_f1",
        "n_concept_slides",
        "n_triple_slides",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for model, stats in global_agg.items():
        cj_count = stats["concept_jaccard_count"]
        tf_count = stats["triple_f1_count"]
        avg_cj = (stats["concept_jaccard_sum"] / cj_count) if cj_count > 0 else ""
        avg_tf1 = (stats["triple_f1_sum"] / tf_count) if tf_count > 0 else ""
        writer.writerow({
            "model": model,
            "global_avg_concept_jaccard": avg_cj,
            "global_avg_triple_f1": avg_tf1,
            "n_concept_slides": cj_count,
            "n_triple_slides": tf_count,
        })

print("Done.")
