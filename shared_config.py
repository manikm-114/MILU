# shared_config.py
import os, datetime

# Root = where this script is run from (Further Work)
ROOT = os.path.abspath(".")

MILU = os.path.join(ROOT, "MILU23")
DATA_DIR = os.path.join(MILU, "data")
BY_SLIDE_DIR = os.path.join(DATA_DIR, "by_slide")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
FUSION_DIR = os.path.join(DATA_DIR, "fusion")
FUSION_PATH = os.path.join(FUSION_DIR, "fusion_multi_models.jsonl")
LOG_PATH = os.path.join(ROOT, "pipeline.log")

for d in [DATA_DIR, BY_SLIDE_DIR, ANALYSIS_DIR, FUSION_DIR]:
    os.makedirs(d, exist_ok=True)

# Final 4 models to use everywhere
SELECTED_MODELS = [
    "llava-hf__llava-onevision-qwen2-7b-ov-hf",
    "OpenGVLab__InternVL3-14B",
    "Qwen__Qwen2-VL-7B-Instruct",
    "Qwen__Qwen3-VL-4B-Instruct",
]

def log_line(script: str, msg: str):
    """Write a line to stdout and pipeline.log with timestamp + script name."""
    ts = datetime.datetime.utcnow().isoformat()
    line = f"[{ts}] [{script}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
