# build_by_slide.py
import os, re, json
from typing import List, Dict, Any

from shared_config import MILU, BY_SLIDE_DIR, SELECTED_MODELS, log_line

SCRIPT = "build_by_slide"

def list_lectures(base: str) -> List[str]:
    out = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if name.lower().startswith("lecture ") and os.path.isdir(p):
            out.append(name)
    out.sort(key=lambda x: int(re.findall(r"\d+", x)[-1]))
    return out

def list_slides(lec_dir: str) -> List[str]:
    """Returns slide index numbers based on Images/SlideXX.*"""
    img_dir = os.path.join(lec_dir, "Images")
    if not os.path.isdir(img_dir):
        return []
    out = []
    for fname in os.listdir(img_dir):
        if fname.lower().startswith("slide") and fname.lower().endswith((".jpg", ".jpeg", ".png")):
            idxs = re.findall(r"\d+", fname)
            if not idxs:
                continue
            out.append(int(idxs[-1]))
    return sorted(set(out))

def load_model_file(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def main():
    total_slides = 0
    lectures = list_lectures(MILU)

    log_line(SCRIPT, f"Building by_slide directory at: {BY_SLIDE_DIR}")

    for lec in lectures:
        lec_dir = os.path.join(MILU, lec)
        slides = list_slides(lec_dir)
        if not slides:
            continue

        out_lec_dir = os.path.join(BY_SLIDE_DIR, lec)
        os.makedirs(out_lec_dir, exist_ok=True)

        log_line(SCRIPT, f"Processing {lec} — {len(slides)} slides")

        for sid in slides:
            slide_id = f"Slide{sid}"
            img_path = os.path.join(lec_dir, "Images", f"{slide_id}.JPG")
            if not os.path.exists(img_path):
                # try png/jpg lowercase fallbacks
                for ext in [".jpg", ".jpeg", ".png"]:
                    alt = os.path.join(lec_dir, "Images", f"{slide_id}{ext}")
                    if os.path.exists(alt):
                        img_path = alt
                        break

            txt_path = os.path.join(lec_dir, "Texts", f"{slide_id}.txt")

            models_block: Dict[str, Any] = {}
            for model in SELECTED_MODELS:
                m_dir = os.path.join(lec_dir, "Outputs", model)
                c_path = os.path.join(m_dir, "concepts", f"{slide_id}.json")
                t_path = os.path.join(m_dir, "triples",  f"{slide_id}.json")

                c_obj = load_model_file(c_path) if os.path.exists(c_path) else None
                t_obj = load_model_file(t_path) if os.path.exists(t_path) else None

                entry: Dict[str, Any] = {}
                if c_obj is not None:
                    entry["concepts"] = {
                        "source": c_path,
                        "parsed": c_obj.get("parsed"),
                        "raw": json.dumps(c_obj, ensure_ascii=False),
                    }
                else:
                    entry["concepts"] = None

                if t_obj is not None:
                    entry["triples"] = {
                        "source": t_path,
                        "parsed": t_obj.get("parsed"),
                        "raw": json.dumps(t_obj, ensure_ascii=False),
                    }
                else:
                    entry["triples"] = None

                models_block[model] = entry

            out_obj = {
                "lecture": lec,
                "slide_id": slide_id,
                "paths": {
                    "image": img_path,
                    "text": txt_path,
                },
                "models": models_block,
            }

            out_path = os.path.join(out_lec_dir, f"{slide_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_obj, f, ensure_ascii=False, indent=2)

            total_slides += 1

    log_line(SCRIPT, f"✅ Done. Built {total_slides} slide JSON files into by_slide/")
    log_line(SCRIPT, f"Location: {BY_SLIDE_DIR}")

if __name__ == "__main__":
    main()
