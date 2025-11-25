# analyze_and_visualize_agreement.py

import os, csv
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

ROOT = os.path.abspath(".")
MILU = os.path.join(ROOT, "MILU23")
DATA_DIR = os.path.join(MILU, "data")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
FIG_DIR = os.path.join(ANALYSIS_DIR, "figures")
LOG_DIR = os.path.join(MILU, "logs")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
PIPELINE_LOG = os.path.join(LOG_DIR, "pipeline.log")


def log(msg: str) -> None:
    print(msg)
    with open(PIPELINE_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main() -> None:
    csv_path = os.path.join(ANALYSIS_DIR, "lecture_level_agreement.csv")
    if not os.path.isfile(csv_path):
        log(f"❌ lecture_level_agreement.csv not found: {csv_path}")
        return

    lecture_data: Dict[int, Dict[str, List[Tuple[float, float]]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lec_idx = int(row["lecture_index"])
            cj = float(row["avg_concept_jaccard"])
            tf1 = float(row["avg_triple_f1"])
            key = f'{row["model_a"]} vs {row["model_b"]}'
            lecture_data.setdefault(lec_idx, {}).setdefault(key, []).append((cj, tf1))

    # aggregate per lecture across all model pairs
    sorted_lectures = sorted(lecture_data.keys())
    x = []
    cj_means = []
    tf1_means = []
    for lec_idx in sorted_lectures:
        all_cj = []
        all_tf1 = []
        for pair, vals in lecture_data[lec_idx].items():
            for cj, tf1 in vals:
                all_cj.append(cj)
                all_tf1.append(tf1)
        x.append(lec_idx)
        cj_means.append(sum(all_cj) / len(all_cj))
        tf1_means.append(sum(all_tf1) / len(all_tf1))

    # Plot two-panel lecture trend
    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(x, cj_means, marker="o")
    ax1.set_title("Lecture-wise Model Agreement (Concepts & Triples)")
    ax1.set_ylabel("Mean Concept Jaccard")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_xticks(x)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(x, tf1_means, marker="o")
    ax2.set_xlabel("Lecture Index")
    ax2.set_ylabel("Mean Triple F1")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_xticks(x)

    fig_path = os.path.join(FIG_DIR, "lecture_trend_plot.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    log(f"✅ Saved lecture trend figure to {fig_path}")


if __name__ == "__main__":
    main()
