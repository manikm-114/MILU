# analyze_model_agreement_multi.py
import os, json, csv, re
from itertools import combinations
from typing import List, Dict, Any, Tuple

from shared_config import BY_SLIDE_DIR, ANALYSIS_DIR, SELECTED_MODELS, log_line
from fuse_models_multi import extract_concepts, extract_triples

SCRIPT = "analyze_model_agreement_multi"

def safe_jload(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def list_lectures(base: str) -> List[str]:
    out = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if name.lower().startswith("lecture ") and os.path.isdir(p):
            out.append(name)
    out.sort(key=lambda x: int(re.findall(r"\d+", x)[-1]))
    return out

def list_slide_jsons(lec_dir: str) -> List[str]:
    return sorted(
        [f for f in os.listdir(lec_dir) if f.lower().endswith(".json")],
        key=lambda x: int(re.findall(r"\d+", x)[-1])
    )

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def triple_f1(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    prec = inter / len(sa) if sa else 0.0
    rec  = inter / len(sb) if sb else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    slide_csv   = os.path.join(ANALYSIS_DIR, "slide_level_agreement.csv")
    lecture_csv = os.path.join(ANALYSIS_DIR, "lecture_level_agreement.csv")
    pair_csv    = os.path.join(ANALYSIS_DIR, "model_pair_overall.csv")

    slide_rows: List[Dict[str, Any]] = []
    total_pairs = 0

    for lec in list_lectures(BY_SLIDE_DIR):
        lec_dir = os.path.join(BY_SLIDE_DIR, lec)
        for sf in list_slide_jsons(lec_dir):
            p = os.path.join(lec_dir, sf)
            j = safe_jload(p)
            if not j or "models" not in j:
                continue
            models = j["models"]

            conc = {}
            trip = {}
            for m in SELECTED_MODELS:
                md = models.get(m, {})
                if not isinstance(md, dict):
                    continue
                conc[m] = extract_concepts(md)
                trip[m] = extract_triples(md)

            for a, b in combinations(SELECTED_MODELS, 2):
                ca, cb = conc.get(a, []), conc.get(b, [])
                ta, tb = trip.get(a, []), trip.get(b, [])

                # --- Filtering (Option C) ---
                # Only compute if both sides have "reasonable" content
                cj = jaccard(ca, cb) if (len(ca) >= 2 and len(cb) >= 2) else 0.0
                tf = triple_f1(ta, tb) if (len(ta) >= 1 and len(tb) >= 1) else 0.0

                slide_rows.append({
                    "lecture": j.get("lecture"),
                    "slide_id": j.get("slide_id"),
                    "model_a": a,
                    "model_b": b,
                    "concept_jaccard": cj,
                    "triple_f1": tf,
                })
                total_pairs += 1

    # --- slide-level CSV ---
    with open(slide_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["lecture", "slide_id", "model_a", "model_b",
                           "concept_jaccard", "triple_f1"]
        )
        w.writeheader()
        for r in slide_rows:
            w.writerow(r)

    # --- lecture-level aggregation ---
    lec_agg: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for r in slide_rows:
        key = (r["lecture"], r["model_a"], r["model_b"])
        lec_agg.setdefault(key, {"sum_cj": 0.0, "sum_tf": 0.0, "n": 0})
        lec_agg[key]["sum_cj"] += float(r["concept_jaccard"])
        lec_agg[key]["sum_tf"] += float(r["triple_f1"])
        lec_agg[key]["n"] += 1

    with open(lecture_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["lecture", "lecture_index",
                        "model_a", "model_b",
                        "avg_concept_jaccard", "avg_triple_f1", "n_slides"]
        )
        w.writeheader()
        for (lec, ma, mb), v in sorted(
            lec_agg.items(),
            key=lambda kv: (int(re.findall(r"\d+", kv[0][0])[-1]), kv[0][1], kv[0][2])
        ):
            n = max(v["n"], 1)
            idx = int(re.findall(r"\d+", lec)[-1])
            w.writerow({
                "lecture": lec,
                "lecture_index": idx,
                "model_a": ma,
                "model_b": mb,
                "avg_concept_jaccard": v["sum_cj"] / n,
                "avg_triple_f1": v["sum_tf"] / n,
                "n_slides": v["n"],
            })

    # --- overall model-pair aggregation (for one main table) ---
    pair_agg: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in slide_rows:
        key = (r["model_a"], r["model_b"])
        pair_agg.setdefault(key, {"sum_cj": 0.0, "sum_tf": 0.0, "n": 0})
        pair_agg[key]["sum_cj"] += float(r["concept_jaccard"])
        pair_agg[key]["sum_tf"] += float(r["triple_f1"])
        pair_agg[key]["n"] += 1

    with open(pair_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_a", "model_b",
                    "avg_concept_jaccard", "avg_triple_f1", "n_slide_pairs"])
        for (ma, mb), v in sorted(pair_agg.items()):
            n = max(v["n"], 1)
            w.writerow([ma, mb, v["sum_cj"]/n, v["sum_tf"]/n, v["n"]])

    log_line(SCRIPT, f"✅ Slide-level saved to: {slide_csv}")
    log_line(SCRIPT, f"✅ Lecture-level saved to: {lecture_csv}")
    log_line(SCRIPT, f"✅ Overall pair summary saved to: {pair_csv}")
    log_line(SCRIPT, f"✅ Total slide pairs analyzed: {total_pairs}")

if __name__ == "__main__":
    main()
