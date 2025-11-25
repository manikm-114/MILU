# report_parsing_coverage.py
import os, re, json, csv
from typing import List

from shared_config import BY_SLIDE_DIR, ANALYSIS_DIR, SELECTED_MODELS, log_line

SCRIPT = "report_parsing_coverage"

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

def safe_jload(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def main():
    if not os.path.isdir(BY_SLIDE_DIR):
        log_line(SCRIPT, f"❌ Not found: {BY_SLIDE_DIR}")
        return

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    cov_csv = os.path.join(ANALYSIS_DIR, "parsing_coverage.csv")

    # model -> lecture -> [have, total]
    per_model_concepts = {m: {} for m in SELECTED_MODELS}
    per_model_triples  = {m: {} for m in SELECTED_MODELS}

    lectures = list_lectures(BY_SLIDE_DIR)
    for lec in lectures:
        lec_dir = os.path.join(BY_SLIDE_DIR, lec)
        slide_files = list_slide_jsons(lec_dir)
        total = len(slide_files)
        if total == 0:
            continue

        for m in SELECTED_MODELS:
            per_model_concepts[m].setdefault(lec, [0, total])
            per_model_triples[m].setdefault(lec, [0, total])

        for sf in slide_files:
            p = os.path.join(lec_dir, sf)
            j = safe_jload(p)
            if not j or "models" not in j:
                continue
            models = j["models"]

            for m in SELECTED_MODELS:
                md = models.get(m, {})
                # Valid only if we actually have a parsed structure (not None)
                c = md.get("concepts")
                t = md.get("triples")

                if isinstance(c, dict) and c.get("parsed") is not None:
                    per_model_concepts[m][lec][0] += 1
                if isinstance(t, dict) and t.get("parsed") is not None:
                    per_model_triples[m][lec][0] += 1

    # Print to console + CSV for table
    with open(cov_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "lecture", "lecture_index",
                         "concepts_valid", "concepts_total",
                         "triples_valid", "triples_total"])

        log_line(SCRIPT, "\n=== PARSING COVERAGE (valid parsed JSON only) ===\n")

        for m in SELECTED_MODELS:
            log_line(SCRIPT, f"[{m}]")
            for lec in lectures:
                cv = per_model_concepts[m].get(lec, [0, 0])
                tv = per_model_triples[m].get(lec, [0, 0])
                c_have, c_tot = cv
                t_have, t_tot = tv
                c_pct = (c_have / c_tot * 100.0) if c_tot else 0.0
                t_pct = (t_have / t_tot * 100.0) if t_tot else 0.0
                idx = int(re.findall(r"\d+", lec)[-1])

                log_line(SCRIPT,
                    f"  {lec:>9}: C {c_have:4d}/{c_tot:<4d} ({c_pct:6.2f}%) | "
                    f"T {t_have:4d}/{t_tot:<4d} ({t_pct:6.2f}%)"
                )
                writer.writerow([m, lec, idx, c_have, c_tot, t_have, t_tot])

            log_line(SCRIPT, "")

    log_line(SCRIPT, f"✅ Coverage CSV saved to: {cov_csv}")

if __name__ == "__main__":
    main()
    