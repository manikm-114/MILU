import os
import json
import csv
import time
from typing import Dict, Any, List
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ===================== CONFIG =====================

INPUT_SLIDES_JSONL = "lecture_slides.jsonl"
OUTPUT_JSONL = "ground_truth_human_reference.jsonl"
OUTPUT_CSV = "ground_truth_human_reference.csv"

# TinyLlama for fast inference
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM_PROMPT = """You are an expert assistant helping extract structured information from lecture slides.
Extract:
1. A list of KEY CONCEPTS mentioned on the slide.
2. RELATION TRIPLES in this structure: { "head": ..., "relation": ..., "tail": ... }.

Respond ONLY as valid JSON:
{
  "concepts": [...],
  "triples": [...]
}
"""

USER_PROMPT_TEMPLATE = "\nSlide:\n{slide_text}\n\n"

# ===================== MODEL LOADING =====================

def load_model():
    print("Loading TinyLlama model with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

# ===================== UTILITIES =====================

def read_slides(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def extract_json(text: str) -> Dict[str, Any]:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"concepts": [], "triples": []}

# ===================== MAIN PROCESSING =====================

def generate_reference(slides, tokenizer, model):
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as jsonl_file, \
         open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(["slide_number", "text", "concepts", "triples", "raw_output"])

        for slide in tqdm(slides, desc="Processing slides"):
            prompt = SYSTEM_PROMPT + USER_PROMPT_TEMPLATE.format(slide_text=slide["content"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs, 
                max_new_tokens=300,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            json_data = extract_json(decoded)

            record = {
                "slide_number": slide["slide_number"],
                "content": slide["content"],
                "concepts": json_data.get("concepts", []),
                "triples": json_data.get("triples", []),
                "raw_output": decoded
            }
            jsonl_file.write(json.dumps(record) + "\n")
            writer.writerow([
                slide["slide_number"], slide["content"],
                json.dumps(record["concepts"]), json.dumps(record["triples"]),
                decoded[:100] + "..."
            ])

# ======================= ENTRYPOINT =======================

if __name__ == "__main__":
    if not os.path.exists(INPUT_SLIDES_JSONL):
        raise FileNotFoundError(f"Input file missing: {INPUT_SLIDES_JSONL}")

    slides = read_slides(INPUT_SLIDES_JSONL)
    print(f"Loaded {len(slides)} slides.")

    tokenizer, model = load_model()
    generate_reference(slides, tokenizer, model)
    print(f"Done! Results saved to {OUTPUT_JSONL} and {OUTPUT_CSV}")
