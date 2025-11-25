# fuse_models_multi.py
import os, json, re
from typing import Dict, Any, List

from shared_config import BY_SLIDE_DIR, FUSION_PATH, SELECTED_MODELS, log_line

SCRIPT = "fuse_models_multi"

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

# ---------- parsing helpers ----------

def extract_concepts(model_dict: Dict[str, Any]) -> List[str]:
    c = model_dict.get("concepts")
    if not isinstance(c, dict):
        return []
    parsed = c.get("parsed")
    if parsed is None:
        return []

    items: List[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        if "concepts" in parsed and isinstance(parsed["concepts"], list):
            items = parsed["concepts"]
        elif "term" in parsed:
            items = [parsed]
    elif isinstance(parsed, list):
        items = parsed

    terms = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        term = obj.get("term")
        if isinstance(term, str) and term.strip():
            terms.append(term.strip().lower())
    return sorted(set(terms))

def canon_triple(t: Dict[str, Any]) -> str:
    s = (t.get("s") or "").strip().lower()
    p = (t.get("p") or "").strip().lower()
    o = (t.get("o") or "").strip().lower()
    return f"{s}||{p}||{o}"

def extract_triples(model_dict: Dict[str, Any]) -> List[str]:
    tr = model_dict.get("triples")
    if not isinstance(tr, dict):
        return []
    parsed = tr.get("parsed")
    if parsed is None:
        return []

    items: List[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        if "triples" in parsed and isinstance(parsed["triples"], list):
            items = parsed["triples"]
        elif {"s", "p", "o"} <= set(parsed.keys()):
            items = [parsed]
    elif isinstance(parsed, list):
        items = parsed

    out = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        key = canon_triple(obj)
        if key.strip("||"):
            out.append(key)
    return sorted(set(out))

# ---------- fusion ----------

def main():
    os.makedirs(os.path.dirname(FUSION_PATH), exist_ok=True)
    total_slides = 0

    with open(FUSION_PATH, "w", encoding="utf-8") as fout:
        for lec in list_lectures(BY_SLIDE_DIR):
            lec_dir = os.path.join(BY_SLIDE_DIR, lec)
            for sf in list_slide_jsons(lec_dir):
                p = os.path.join(lec_dir, sf)
                j = safe_jload(p)
                if not j or "models" not in j:
                    continue
                models = j["models"]

                parsed_conc = {}
                parsed_trip = {}

                for m in SELECTED_MODELS:
                    md = models.get(m, {})
                    if not isinstance(md, dict):
                        continue
                    parsed_conc[m] = extract_concepts(md)
                    parsed_trip[m] = extract_triples(md)

                # consensus (>=2 models must agree)
                concept_counts = {}
                triple_counts = {}

                for cset in parsed_conc.values():
                    for c in cset:
                        concept_counts[c] = concept_counts.get(c, 0) + 1

                for tset in parsed_trip.values():
                    for t in tset:
                        triple_counts[t] = triple_counts.get(t, 0) + 1

                fused_concepts = sorted([c for c, n in concept_counts.items() if n >= 2])
                fused_triples = sorted([t for t, n in triple_counts.items() if n >= 2])

                out_rec = {
                    "lecture": j.get("lecture"),
                    "slide_id": j.get("slide_id"),
                    "paths": j.get("paths", {}),
                    "models": parsed_conc,
                    "triples": parsed_trip,
                    "superlearner": {
                        "concepts": fused_concepts,
                        "triples": fused_triples,
                    },
                }
                fout.write(json.dumps(out_rec) + "\n")
                total_slides += 1

    log_line(SCRIPT, f"✅ Fused {total_slides} slides")
    log_line(SCRIPT, f"File saved: {FUSION_PATH}")

if __name__ == "__main__":
    main()
