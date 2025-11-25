# superlearner_fusion.py
import os, json
from collections import Counter, defaultdict
from tqdm import tqdm

ROOT = os.path.join("MILU23", "data", "by_slide")
OUT_DIR = os.path.join("MILU23", "data", "superlearner")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    "OpenGVLab__InternVL3-8B",
    "OpenGVLab__InternVL3-14B",
    "Qwen__Qwen2-VL-7B-Instruct",
    "Qwen__Qwen3-VL-4B-Instruct",
]

def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def unique_key_concept(c):
    return (c.get("term", "").lower().strip(), c.get("category", "").lower().strip())

def unique_key_triple(t):
    return (
        t.get("s", "").lower().strip(),
        t.get("p", "").lower().strip(),
        t.get("o", "").lower().strip(),
    )

def fuse_slide(model_data_list):
    """Combine outputs from multiple models by majority vote."""
    all_concepts, all_triples = [], []
    for d in model_data_list:
        if not d or "parsed" not in d: 
            continue
        parsed = d["parsed"]
        if isinstance(parsed, dict):
            if "concepts" in parsed and isinstance(parsed["concepts"], list):
                all_concepts.extend(parsed["concepts"])
            if "triples" in parsed and isinstance(parsed["triples"], list):
                all_triples.extend(parsed["triples"])

    fused = {"concepts": [], "triples": []}

    # Majority vote for concepts
    term_counts = Counter(unique_key_concept(c) for c in all_concepts if c)
    for key, count in term_counts.items():
        if count >= 2:
            fused["concepts"].append({"term": key[0], "category": key[1]})

    # Majority vote for triples
    triple_counts = Counter(unique_key_triple(t) for t in all_triples if t)
    for key, count in triple_counts.items():
        if count >= 2:
            fused["triples"].append({"s": key[0], "p": key[1], "o": key[2]})

    return fused

def run():
    total_slides = 0
    for lecture in sorted(os.listdir(ROOT)):
        lec_path = os.path.join(ROOT, lecture)
        if not os.path.isdir(lec_path):
            continue
        slides = [f for f in os.listdir(lec_path) if f.endswith(".json")]
        out_lec = os.path.join(OUT_DIR, lecture)
        os.makedirs(out_lec, exist_ok=True)

        for slide_file in tqdm(slides, desc=lecture):
            total_slides += 1
            slide_path = os.path.join(lec_path, slide_file)
            # each slide file should contain per-model outputs under parsed
            data = read_json(slide_path)
            if not data or "models" not in data:
                continue

            model_data = [data["models"].get(m) for m in MODELS if m in data["models"]]
            fused = fuse_slide(model_data)

            out_path = os.path.join(out_lec, slide_file)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"slide_id": slide_file.replace(".json", ""), "superlearner": fused}, f, indent=2)

    print(f"✅ Superlearner fusion complete for {total_slides} slides.")
    print(f"✅ Outputs saved in: {OUT_DIR}")

if __name__ == "__main__":
    run()
