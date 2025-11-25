# evaluate_superlearner.py
import os, json, csv, re
from typing import Dict, Any, List

from shared_config import FUSION_PATH, ANALYSIS_DIR, SELECTED_MODELS, log_line
from analyze_model_agreement_multi import jaccard, triple_f1

SCRIPT = "evaluate_superlearner"

def safe_jloadl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def main():
    if not os.path.isfile(FUSION_PATH):
        log_line(SCRIPT, f"Fusion file not found: {FUSION_PATH}")
        return

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    slide_csv = os.path.join(ANALYSIS_DIR, "superlearner_evaluation.csv")
    lec_csv   = os.path.join(ANALYSIS_DIR, "superlearner_evaluation_lecture.csv")
    overall_csv = os.path.join(ANALYSIS_DIR, "superlearner_overall.csv")

    rows = []

    for rec in safe_jloadl(FUSION_PATH):
        lec = rec.get("lecture")
        slide_id = rec.get("slide_id")

        sl_conc = list(sorted(set(rec.get("superlearner", {}).get("concepts", []) or [])))
        sl_trip = list(sorted(set(rec.get("superlearner", {}).get("triples", []) or [])))

        models_conc: Dict[str, List[str]] = rec.get("models", {}) or {}
        models_trip: Dict[str, List[str]] = rec.get("triples", {}) or {}

        for m in SELECTED_MODELS:
            mc = models_conc.get(m, []) or []
            mt = models_trip.get(m, []) or []

            # filtering: same logic as earlier
            cj = jaccard(sl_conc, mc) if (len(sl_conc) >= 2 and len(mc) >= 2) else 0.0
            tf = triple_f1(sl_trip, mt) if (len(sl_trip) >= 1 and len(mt) >= 1) else 0.0

            rows.append({
                "lecture": lec,
                "slide_id": slide_id,
                "model": m,
                "concept_jaccard": cj,
                "triple_f1": tf,
            })

    # slide-level CSV
    with open(slide_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["lecture", "slide_id", "model",
                           "concept_jaccard", "triple_f1"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # overall + lecture-wise aggregation
    overall = {}
    per_lecture = {}

    for r in rows:
        m = r["model"]
        overall.setdefault(m, {"sum_cj": 0.0, "sum_tf": 0.0, "n": 0})
        overall[m]["sum_cj"] += float(r["concept_jaccard"])
        overall[m]["sum_tf"] += float(r["triple_f1"])
        overall[m]["n"] += 1

        key = (r["lecture"], m)
        per_lecture.setdefault(key, {"sum_cj": 0.0, "sum_tf": 0.0, "n": 0})
        per_lecture[key]["sum_cj"] += float(r["concept_jaccard"])
        per_lecture[key]["sum_tf"] += float(r["triple_f1"])
        per_lecture[key]["n"] += 1

    # overall CSV (nice for one key table)
    with open(overall_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "mean_concept_jaccard", "mean_triple_f1", "n_slide_evals"])
        for m, v in overall.items():
            n = max(v["n"], 1)
            cj = v["sum_cj"]/n
            tf = v["sum_tf"]/n
            w.writerow([m, cj, tf, v["n"]])
            log_line(SCRIPT, f"{m:35s} concept_jaccard={cj:.3f} triple_f1={tf:.3f}")

    # lecture-level CSV
    with open(lec_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lecture", "lecture_index", "model",
                    "mean_concept_jaccard", "mean_triple_f1", "n_slides"])
        for (lec, m), v in sorted(
            per_lecture.items(),
            key=lambda kv: (int(re.findall(r"\d+", kv[0][0])[-1]), kv[0][1])
        ):
            n = max(v["n"], 1)
            idx = int(re.findall(r"\d+", lec)[-1])
            w.writerow([lec, idx, m,
                        v["sum_cj"]/n,
                        v["sum_tf"]/n,
                        v["n"]])

    log_line(SCRIPT, f"✅ Superlearner evaluation written to: {slide_csv}, {lec_csv}, {overall_csv}")

if __name__ == "__main__":
    main()
