"""
compare_models.py
=================
Gathers the results of all four algorithms into one table and the charts
used in the PowerPoint report.

HOW TO USE
----------
Option A (recommended): run the four model scripts first -
    python knn_model.py
    python decision_tree_model.py
    python ann_model.py
    python svm_model.py
then run:
    python compare_models.py

Option B: just run this script. If a model's results file is missing it
will run that model itself (the SVM step takes several minutes).

WHAT IT PRODUCES (all inside the parent folder's Results/)
----------------------------------------------------------
* Results/summary.csv                    - one row per model, key numbers
* Results/figures/accuracy_comparison.png - CV vs all-samples accuracy
* Results/figures/class_balance.png       - why accuracy alone misleads
* Results/figures/confusion_matrices.png  - where each model errs

READING THE MAIN CHART
----------------------
For each model we plot two bars:
* "2-fold CV" - the honest score: every sample was predicted by a model
  that never saw it during training.
* "All samples" - train and test on the same data: a deliberately
  optimistic score. The GAP between the bars is memorization; a model
  with a huge gap (typically KNN and unpruned trees) is overfitting.
The dashed line is the majority-class baseline (~86%): a model that
always answers "No account" scores this without learning anything, so
only performance ABOVE the line represents real learning.
"""

import json

import matplotlib
matplotlib.use("Agg")  # draw to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_data import RESULTS_DIR, evaluate_model

# Charts go in a subfolder of Results, next to the JSON and CSV outputs.
FIGURES_DIR = RESULTS_DIR / "figures"

# The five models, in the fixed order used everywhere in the report.
MODELS = ["KNN", "Decision Tree", "ANN", "SVM", "LDA"]

# ---------------------------------------------------------------------------
# Chart colors (a colorblind-safe palette, validated for adjacent use)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"        # series 1: the honest 2-fold CV accuracy
ORANGE = "#eb6834"      # series 2: the optimistic all-samples accuracy
INK = "#0b0b0b"         # primary text
MUTED = "#898781"       # axis labels, baseline
GRID = "#e1e0d9"        # hairline gridlines
SURFACE = "#fcfcfb"     # chart background
# Light-to-dark blues for the confusion-matrix heat maps.
BLUES = ["#f3f8fe", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]


def get_results(model_name):
    """Load a model's saved JSON results, running the model if needed."""
    path = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}.json"
    if not path.exists():
        print(f"No saved results for {model_name} - running it now...")
        # Import here (not at the top) so simply loading this script
        # doesn't drag in every model module.
        if model_name == "KNN":
            from knn_model import build_model
        elif model_name == "Decision Tree":
            from decision_tree_model import build_model
        elif model_name == "ANN":
            from ann_model import build_model
        elif model_name == "SVM":
            from svm_model import build_model
        else:
            from lda_model import build_model
        evaluate_model(model_name, build_model())
    with open(path) as f:
        return json.load(f)


def style_axis(ax):
    """Shared look: quiet grid and axes so the data stands out."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)  # gridlines behind the bars
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)


def chart_accuracy_comparison(results):
    """Grouped bars: honest CV accuracy vs optimistic all-samples accuracy."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)

    x = np.arange(len(MODELS))
    width = 0.38
    cv_vals = [results[m]["cv_accuracy"] for m in MODELS]
    all_vals = [results[m]["all_samples_accuracy"] for m in MODELS]

    ax.bar(x - width / 2, cv_vals, width, color=BLUE,
           label="2-fold cross-validation", zorder=3)
    ax.bar(x + width / 2, all_vals, width, color=ORANGE,
           label="All samples (train = test)", zorder=3)

    # Direct value labels on each bar, in text ink (never the bar color).
    for xpos, val in list(zip(x - width / 2, cv_vals)) + \
                     list(zip(x + width / 2, all_vals)):
        ax.text(xpos, val + 0.012, f"{val:.3f}", ha="center",
                fontsize=9, color=INK)

    # The no-skill baseline every model must beat (named in the legend).
    baseline = results["KNN"]["baseline_accuracy"]
    ax.axhline(baseline, color=MUTED, linewidth=1.2, linestyle="--", zorder=2,
               label=f"Always-'No' baseline ({baseline:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11, color=INK)
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Classification accuracy", fontsize=11, color=INK)
    ax.set_title("Accuracy by model: honest (CV) vs optimistic (all samples)",
                 fontsize=13, color=INK, pad=14)
    ax.legend(frameon=False, fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_comparison.png",
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def chart_class_balance():
    """Show the Yes/No imbalance that shapes how we read every accuracy."""
    from prepare_data import load_features_and_target
    _, y = load_features_and_target()
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    labels = ["No account", "Has account"]

    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)

    bars = ax.bar(labels, counts, width=0.55, color=BLUE, zorder=3)
    for bar, count in zip(bars, counts):
        share = count / sum(counts)
        ax.text(bar.get_x() + bar.get_width() / 2, count + 350,
                f"{count:,}  ({share:.0%})", ha="center",
                fontsize=11, color=INK)

    ax.set_ylim(0, 23500)
    ax.set_ylabel("Respondents", fontsize=11, color=INK)
    ax.set_title("The classes are imbalanced: only 14% have a bank account",
                 fontsize=13, color=INK, pad=14)
    ax.tick_params(axis="x", labelsize=11, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png",
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def chart_confusion_matrices(results):
    """One 2x2 confusion matrix per model, from the CV predictions."""
    # One column per model, sized so any number of models fits neatly.
    fig, axes = plt.subplots(1, len(MODELS), figsize=(2.6 * len(MODELS), 3.6),
                             dpi=200)
    fig.patch.set_facecolor(SURFACE)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)

    for ax, model_name in zip(axes, MODELS):
        cm = np.array(results[model_name]["confusion_matrix"])
        # Shade each row by its own share so both classes are visible
        # despite the 6:1 imbalance in row totals.
        row_share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1)

        for i in range(2):
            for j in range(2):
                # White text on dark cells, ink on light cells.
                color = "white" if row_share[i, j] > 0.55 else INK
                ax.text(j, i, f"{cm[i, j]:,}\n({row_share[i, j]:.0%})",
                        ha="center", va="center", fontsize=9, color=color)

        ax.set_title(model_name, fontsize=11, color=INK, pad=8)
        ax.set_xticks([0, 1], ["Pred. No", "Pred. Yes"],
                      fontsize=8.5, color=MUTED)
        ax.set_yticks([0, 1], ["True No", "True Yes"],
                      fontsize=8.5, color=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Where each model errs (2-fold CV predictions, all samples; "
                 "cells shaded by share of the true class)",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png",
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    # 1. Collect every model's results (running any that are missing).
    results = {name: get_results(name) for name in MODELS}

    # 2. Build the one-row-per-model summary table.
    rows = []
    for name in MODELS:
        r = results[name]
        rows.append({
            "Model": name,
            "Fold 1 accuracy": r["fold_accuracies"][0],
            "Fold 2 accuracy": r["fold_accuracies"][1],
            "2-fold CV accuracy": r["cv_accuracy"],
            "All-samples accuracy": r["all_samples_accuracy"],
            "Overfit gap": round(r["all_samples_accuracy"]
                                 - r["cv_accuracy"], 4),
            "CV time (s)": r["cv_seconds"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print("\nSUMMARY (also saved to results/summary.csv)\n")
    print(summary.to_string(index=False))

    # 3. Draw the three report charts.
    chart_accuracy_comparison(results)
    chart_class_balance()
    chart_confusion_matrices(results)
    print(f"\nCharts saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()