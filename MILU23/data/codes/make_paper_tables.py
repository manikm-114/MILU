import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "MILU23", "data", "analysis")
os.makedirs(DATA_DIR, exist_ok=True)

slide_csv = os.path.join(DATA_DIR, "slide_level_agreement.csv")
lecture_csv = os.path.join(DATA_DIR, "lecture_level_agreement.csv")
super_csv = os.path.join(DATA_DIR, "superlearner_evaluation.csv")

# --- 1. MODEL-PAIR SUMMARY (ACROSS ALL SLIDES) ---

df_slides = pd.read_csv(slide_csv)

pair_summary = (
    df_slides
    .groupby(["model_a", "model_b"], as_index=False)[["concept_jaccard", "triple_f1"]]
    .mean()
    .sort_values("concept_jaccard", ascending=False)
)

pair_out = os.path.join(DATA_DIR, "table_model_pair_summary.csv")
pair_summary.to_csv(pair_out, index=False)
print(f"✅ Saved model-pair summary to: {pair_out}")

# --- 2. LECTURE-LEVEL SUMMARY (ALREADY AGGREGATED BUT WE CLEAN IT) ---

df_lecture = pd.read_csv(lecture_csv)

# Just sort nicely by lecture name
df_lecture_sorted = df_lecture.sort_values(["lecture"])
lecture_out = os.path.join(DATA_DIR, "table_lecture_summary.csv")
df_lecture_sorted.to_csv(lecture_out, index=False)
print(f"✅ Saved lecture summary to: {lecture_out}")

# --- 3. SUPERLEARNER SUMMARY ---

df_super = pd.read_csv(super_csv)

# Ensure nice sorting by concept_jaccard (descending)
df_super_sorted = (
    df_super
    .sort_values("concept_jaccard", ascending=False)
    .reset_index(drop=True)
)

super_out = os.path.join(DATA_DIR, "table_superlearner_summary.csv")
df_super_sorted.to_csv(super_out, index=False)
print(f"✅ Saved superlearner summary to: {super_out}")

print("\n=== Quick previews ===")
print("\n[Model-pair summary]")
print(pair_summary)

print("\n[Superlearner summary]")
print(df_super_sorted)
