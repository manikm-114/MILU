# summarize_cross_model_concepts.py
import os, csv, re
from collections import defaultdict

from shared_config import ANALYSIS_DIR, log_line

SCRIPT = "summarize_cross_model_concepts"

def main():
    slide_csv = os.path.join(ANALYSIS_DIR, "slide_level_agreement.csv")
    out_csv   = os.path.join(ANALYSIS_DIR, "cross_model_concept_overlap.csv")

    if not os.path.isfile(slide_csv):
        log_line(SCRIPT, f"Missing slide-level file: {slide_csv}")
        return

    agg = defaultdict(lambda: {"sum_cj": 0.0, "n": 0})

    with open(slide_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            lec = r["lecture"]
            cj = float(r["concept_jaccard"])
            if cj <= 0:
                continue
            agg[lec]["sum_cj"] += cj
            agg[lec]["n"] += 1

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lecture", "lecture_index", "mean_concept_jaccard", "n_pairs"])
        for lec in sorted(agg.keys(), key=lambda x: int(re.findall(r"\d+", x)[-1])):
            v = agg[lec]
            n = max(v["n"], 1)
            idx = int(re.findall(r"\d+", lec)[-1])
            w.writerow([lec, idx, v["sum_cj"]/n, v["n"]])

    log_line(SCRIPT, f"✅ Saved cross-model concept overlap to: {out_csv}")

if __name__ == "__main__":
    main()
