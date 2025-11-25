# advanced_visualizations.py
import os, csv, re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from shared_config import ANALYSIS_DIR, SELECTED_MODELS, log_line

SCRIPT = "advanced_visualizations"

# -------------------------------------------------------------
# Helper: read CSV
# -------------------------------------------------------------
def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():

    # =============================================================
    # 1) Parsing Coverage (Concepts + Triples)
    # =============================================================
    cov_path = os.path.join(ANALYSIS_DIR, "parsing_coverage.csv")
    if os.path.isfile(cov_path):
        rows = read_csv(cov_path)

        agg = {m: {"c_have": 0, "c_tot": 0, "t_have": 0, "t_tot": 0}
               for m in SELECTED_MODELS}

        for r in rows:
            m = r["model"]
            if m not in agg:
                continue
            agg[m]["c_have"] += int(r["concepts_valid"])
            agg[m]["c_tot"]  += int(r["concepts_total"])
            agg[m]["t_have"] += int(r["triples_valid"])
            agg[m]["t_tot"]  += int(r["triples_total"])

        models = SELECTED_MODELS
        x = np.arange(len(models))
        width = 0.35

        conc_pct = [agg[m]["c_have"]/agg[m]["c_tot"] if agg[m]["c_tot"] else 0 for m in models]
        trip_pct = [agg[m]["t_have"]/agg[m]["t_tot"] if agg[m]["t_tot"] else 0 for m in models]

        plt.figure(figsize=(8, 4))
        plt.bar(x - width/2, conc_pct, width, label="Concepts")
        plt.bar(x + width/2, trip_pct, width, label="Triples")
        plt.xticks(x, [m.replace("__", "\n") for m in models], rotation=0)
        plt.ylim(0, 1.05)
        plt.ylabel("Valid Parsed Fraction")
        plt.title("Parsing Coverage Across Models")
        plt.legend()
        plt.tight_layout()

        out = os.path.join(ANALYSIS_DIR, "fig_parsing_coverage.png")
        plt.savefig(out, dpi=300)
        plt.close()
        log_line(SCRIPT, f"Saved {out}")


    # =============================================================
    # 2) Cross-Model Concept Overlap (Bar by Lecture)
    # =============================================================
    overlap_csv = os.path.join(ANALYSIS_DIR, "cross_model_concept_overlap.csv")
    if os.path.isfile(overlap_csv):

        rows = read_csv(overlap_csv)
        rows_sorted = sorted(rows, key=lambda r: int(r["lecture_index"]))

        xs = [int(r["lecture_index"]) for r in rows_sorted]
        ys = [float(r["mean_concept_jaccard"]) for r in rows_sorted]

        plt.figure(figsize=(10, 4))
        plt.bar(xs, ys)
        plt.xlabel("Lecture Index")
        plt.ylabel("Mean Concept Overlap (Jaccard)")
        plt.title("Cross-Model Concept Overlap by Lecture")
        plt.xticks(xs)
        plt.tight_layout()

        out = os.path.join(ANALYSIS_DIR, "fig_cross_model_concept_overlap.png")
        plt.savefig(out, dpi=300)
        plt.close()
        log_line(SCRIPT, f"Saved {out}")


    # =============================================================
    # 3) Lecture-wise Trend (Mean Agreement)
    # =============================================================
    lec_agree = os.path.join(ANALYSIS_DIR, "lecture_level_agreement.csv")
    if os.path.isfile(lec_agree):

        rows = read_csv(lec_agree)
        agg = defaultdict(lambda: {"sum_cj": 0.0, "sum_tf": 0.0, "n": 0})

        for r in rows:
            lec = r["lecture"]
            agg[lec]["sum_cj"] += float(r["avg_concept_jaccard"])
            agg[lec]["sum_tf"] += float(r["avg_triple_f1"])
            agg[lec]["n"] += 1

        lecs = sorted(agg.keys(), key=lambda x: int(re.findall(r"\d+", x)[-1]))

        xs = [int(re.findall(r"\d+", l)[-1]) for l in lecs]
        mean_cj = [agg[l]["sum_cj"]/max(agg[l]["n"], 1) for l in lecs]
        mean_tf = [agg[l]["sum_tf"]/max(agg[l]["n"], 1) for l in lecs]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        axes[0].plot(xs, mean_cj, marker="o")
        axes[0].set_ylabel("Mean Concept Jaccard")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        axes[1].plot(xs, mean_tf, marker="o")
        axes[1].set_ylabel("Mean Triple F1")
        axes[1].set_xlabel("Lecture Index")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()

        out = os.path.join(ANALYSIS_DIR, "fig_lecture_trend_concepts_triples.png")
        fig.savefig(out, dpi=300)
        plt.close(fig)
        log_line(SCRIPT, f"Saved {out}")


    # =============================================================
    # 4) Model-Pair Heatmaps (Concept & Triple)
    # =============================================================
    if os.path.isfile(lec_agree):

        rows = read_csv(lec_agree)
        pair_agg = defaultdict(lambda: {"sum_cj": 0.0, "sum_tf": 0.0, "n": 0})

        for r in rows:
            key = (r["model_a"], r["model_b"])
            pair_agg[key]["sum_cj"] += float(r["avg_concept_jaccard"])
            pair_agg[key]["sum_tf"] += float(r["avg_triple_f1"])
            pair_agg[key]["n"] += 1

        models = SELECTED_MODELS
        idx = {m: i for i, m in enumerate(models)}
        n = len(models)

        mat_cj = np.zeros((n, n))
        mat_tf = np.zeros((n, n))

        for (a, b), v in pair_agg.items():
            i, j = idx[a], idx[b]
            cnt = max(v["n"], 1)
            cj = v["sum_cj"]/cnt
            tf = v["sum_tf"]/cnt
            mat_cj[i, j] = mat_cj[j, i] = cj
            mat_tf[i, j] = mat_tf[j, i] = tf

        for mat, name, title in [
            (mat_cj, "concept", "Mean Concept Jaccard Across Model Pairs"),
            (mat_tf, "triple",  "Mean Triple F1 Across Model Pairs"),
        ]:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(mat, interpolation="nearest", aspect="auto")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            labels = [m.replace("__", "\n") for m in models]
            plt.xticks(range(n), labels, rotation=45, ha="right")
            plt.yticks(range(n), labels)
            plt.title(title)
            plt.tight_layout()

            out = os.path.join(ANALYSIS_DIR, f"fig_model_pair_heatmap_{name}.png")
            plt.savefig(out, dpi=300)
            plt.close()
            log_line(SCRIPT, f"Saved {out}")


    # =============================================================
    # 5) Histograms (Per-Slide Concept Jaccard & Triple F1)
    # =============================================================
    slide_agree = os.path.join(ANALYSIS_DIR, "slide_level_agreement.csv")
    if os.path.isfile(slide_agree):

        rows = read_csv(slide_agree)

        cj_vals = [float(r["concept_jaccard"]) for r in rows if float(r["concept_jaccard"]) > 0]
        tf_vals = [float(r["triple_f1"]) for r in rows if float(r["triple_f1"]) > 0]

        if cj_vals:
            plt.figure(figsize=(6, 4))
            plt.hist(cj_vals, bins=20)
            plt.xlabel("Concept Jaccard")
            plt.ylabel("Count")
            plt.title("Distribution of Per-Slide Concept Agreement")
            plt.tight_layout()

            out = os.path.join(ANALYSIS_DIR, "fig_hist_concept_jaccard.png")
            plt.savefig(out, dpi=300)
            plt.close()
            log_line(SCRIPT, f"Saved {out}")

        if tf_vals:
            plt.figure(figsize=(6, 4))
            plt.hist(tf_vals, bins=20)
            plt.xlabel("Triple F1")
            plt.ylabel("Count")
            plt.title("Distribution of Per-Slide Triple Agreement")
            plt.tight_layout()

            out = os.path.join(ANALYSIS_DIR, "fig_hist_triple_f1.png")
            plt.savefig(out, dpi=300)
            plt.close()
            log_line(SCRIPT, f"Saved {out}")


    # =============================================================
    # 6) Consensus Ensemble vs Models (Lecture-Wise Trends)
    # =============================================================
    sl_lec = os.path.join(ANALYSIS_DIR, "superlearner_evaluation_lecture.csv")
    # NOTE: We're reusing this CSV, but renaming the figure titles/output.
    # The CSV stores consensus-vs-model agreement.

    if os.path.isfile(sl_lec):

        rows = read_csv(sl_lec)

        by_model = defaultdict(list)
        for r in rows:
            m = r["model"]
            idx = int(r["lecture_index"])
            cj = float(r["mean_concept_jaccard"])
            tf = float(r["mean_triple_f1"])
            by_model[m].append((idx, cj, tf))

        # -------- Concept Jaccard --------
        plt.figure(figsize=(10, 4))
        for m, vals in by_model.items():
            vals = sorted(vals, key=lambda x: x[0])
            xs = [v[0] for v in vals]
            ys = [v[1] for v in vals]
            plt.plot(xs, ys, marker="o", label=m)

        plt.xlabel("Lecture Index")
        plt.ylabel("Consensus Ensemble vs Model Concept Jaccard")
        plt.title("Consensus Ensemble Agreement with Each Model (Concepts)")
        plt.legend()
        plt.tight_layout()

        out = os.path.join(ANALYSIS_DIR, "fig_consensus_concepts_trend.png")
        plt.savefig(out, dpi=300)
        plt.close()
        log_line(SCRIPT, f"Saved {out}")

        # -------- Triple F1 --------
        plt.figure(figsize=(10, 4))
        for m, vals in by_model.items():
            vals = sorted(vals, key=lambda x: x[0])
            xs = [v[0] for v in vals]
            ys = [v[2] for v in vals]
            plt.plot(xs, ys, marker="o", label=m)

        plt.xlabel("Lecture Index")
        plt.ylabel("Consensus Ensemble vs Model Triple F1")
        plt.title("Consensus Ensemble Agreement with Each Model (Triples)")
        plt.legend()
        plt.tight_layout()

        out = os.path.join(ANALYSIS_DIR, "fig_consensus_triples_trend.png")
        plt.savefig(out, dpi=300)
        plt.close()
        log_line(SCRIPT, f"Saved {out}")

    log_line(SCRIPT, "✅ Advanced visualizations completed.")


# -------------------------------------------------------------
if __name__ == "__main__":
    main()
