from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "figures"
SCET_RESULTS = PROJECT / "results" / "ipm_scet_full_seed13_portfolio_v2" / "scet_summary.csv"
SCET_STRESS = PROJECT / "results" / "ipm_scet_full_seed13_portfolio_v2" / "scet_stress_summary.csv"
RAG_UTILITY_RESULTS = PROJECT / "results" / "ipm_rag_utility_full_seed13_portfolio_v2" / "rag_utility_summary.csv"
RAG_UTILITY_STRESS = PROJECT / "results" / "ipm_rag_utility_full_seed13_portfolio_v2" / "rag_utility_stress_summary.csv"

EXPORT_EXTENSIONS = (".pdf", ".svg", ".png", ".tif")
DATASET_ORDER = ["scifact", "fever", "hover"]
DATASET_LABEL = {"scifact": "SciFact", "fever": "FEVER", "hover": "HoVer"}
METHOD_LABEL = {
    "bm25": "BM25",
    "tfidf_svd": "TF-IDF/SVD",
    "dense": "Dense",
    "mvec_tg": "MVEC-TG",
    "sbert_mini_lm": "SBERT",
    "e5_small_v2": "E5-small",
    "mvec_tg_semantic": "MVEC-TG semantic",
    "bge_m3": "BGE-M3",
    "rag_utility": "RAG utility",
    "scet": "SCET",
}
EVIDENCE_LIST_METHODS = {"bge_m3", "rag_utility", "mvec_tg_semantic"}

PALETTE = {
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#D55E00",
    "purple": "#CC79A7",
    "ink": "#1F2937",
    "grid": "#D1D5DB",
    "light": "#F8FAFC",
    "light_blue": "#EAF4FB",
    "light_green": "#EDF8F3",
    "light_orange": "#FFF3E3",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 7.3,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.titlesize": 7.6,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing figure source data: {path}")
    return pd.read_csv(path)


def save_bundle(fig: plt.Figure, stem: str, *, dpi: int = 600) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for extension in EXPORT_EXTENSIONS:
        out = FIG_DIR / f"{stem}{extension}"
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.025, "facecolor": "white"}
        if extension in {".png", ".tif"}:
            save_kwargs["dpi"] = dpi
        if extension == ".tif":
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(out, **save_kwargs)
        outputs.append(out)
    return outputs


def finish_axis(ax: plt.Axes, *, y_grid: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["ink"])
        spine.set_linewidth(0.7)
    ax.tick_params(direction="out", top=False, right=False, length=3.0, width=0.65, colors="black")
    if y_grid:
        ax.grid(True, axis="y", linewidth=0.35, alpha=0.55, color=PALETTE["grid"])
    else:
        ax.grid(False)


def add_box(ax: plt.Axes, x: float, y: float, width: float, height: float, title: str, body: str, fill: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.75,
        edgecolor=PALETTE["ink"],
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.69, title, ha="center", va="center", weight="bold", color=PALETTE["ink"])
    ax.text(x + width / 2, y + height * 0.33, body, ha="center", va="center", fontsize=6.1, linespacing=1.05, color="#374151")


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9.0,
        linewidth=0.85,
        color=PALETTE["ink"],
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def make_mechanism_figure() -> list[Path]:
    set_style()
    fig, ax = plt.subplots(figsize=(7.1, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.02, 0.64, 0.16, 0.24, "Structured queue", "aircraft\nrotations", PALETTE["light"]),
        (0.02, 0.23, 0.16, 0.24, "Text claim queue", "claims and\nmulti-hop facts", PALETTE["light"]),
        (0.24, 0.64, 0.16, 0.24, "Structured evidence", "cases,\nrules,\nrelations", PALETTE["light_blue"]),
        (0.24, 0.23, 0.16, 0.24, "Text evidence", "sentences,\ntitles,\nsources", PALETTE["light_blue"]),
        (0.47, 0.43, 0.15, 0.28, "Unified schema", "query,\ncandidate,\nsupport label", PALETTE["light_green"]),
        (0.68, 0.43, 0.13, 0.28, "Calibrated support", "views,\nconflict,\nthreshold", PALETTE["light_orange"]),
        (0.86, 0.43, 0.12, 0.28, "Certificate", "evidence,\nscore,\nconflict", "#F4F0FA"),
    ]
    for box in boxes:
        add_box(ax, *box)

    arrows = [
        ((0.18, 0.76), (0.24, 0.76)),
        ((0.18, 0.35), (0.24, 0.35)),
        ((0.40, 0.76), (0.47, 0.59)),
        ((0.40, 0.35), (0.47, 0.55)),
        ((0.62, 0.57), (0.68, 0.57)),
        ((0.81, 0.57), (0.86, 0.57)),
    ]
    for start, end in arrows:
        add_arrow(ax, start, end)

    ax.text(0.64, 0.23, "conflict control reduces high-similarity unsupported evidence", ha="center", va="center", fontsize=6.6, color=PALETTE["ink"])
    add_arrow(ax, (0.74, 0.43), (0.61, 0.30))
    add_arrow(ax, (0.61, 0.30), (0.56, 0.43))

    outputs = save_bundle(fig, "ipm_scet_mechanism")
    plt.close(fig)
    return outputs


def best_comparison_rows(summary: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        dataset_rows = summary[summary["dataset"].eq(dataset)].copy()
        stress_rows = stress[stress["dataset"].eq(dataset)].copy()
        scet = dataset_rows[dataset_rows["retriever"].eq("scet")].iloc[0]
        scet_stress = stress_rows[stress_rows["retriever"].eq("scet")].iloc[0]

        comparisons = dataset_rows[~dataset_rows["retriever"].eq("scet")].copy()
        stress_comparisons = stress_rows[~stress_rows["retriever"].eq("scet")].copy()
        best_precision = comparisons.sort_values("support_precision", ascending=False).iloc[0]
        best_recall = comparisons.sort_values("recall_at_5", ascending=False).iloc[0]
        best_stress = stress_comparisons.sort_values("stress_support_precision", ascending=False).iloc[0]

        rows.append(
            {
                "dataset": dataset,
                "label": DATASET_LABEL[dataset],
                "comparison_precision_method": METHOD_LABEL.get(best_precision["retriever"], best_precision["retriever"]),
                "comparison_precision": float(best_precision["support_precision"]),
                "scet_precision": float(scet["support_precision"]),
                "comparison_stress_method": METHOD_LABEL.get(best_stress["retriever"], best_stress["retriever"]),
                "comparison_stress": float(best_stress["stress_support_precision"]),
                "scet_stress": float(scet_stress["stress_support_precision"]),
                "recall_delta_vs_best": float(scet["recall_at_5"]) - float(best_recall["recall_at_5"]),
            }
        )
    return pd.DataFrame(rows)


def add_pair_labels(ax: plt.Axes, bars, values: list[float], *, percent: bool = False) -> None:
    ymax = max(max(values), 0.01)
    offset = ymax * 0.025
    for bar, value in zip(bars, values):
        text = f"{value * 100:.1f}" if percent else f"{value:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, text, ha="center", va="bottom", fontsize=5.9)


def make_results_figure() -> list[Path]:
    set_style()
    summary = read_csv_required(SCET_RESULTS)
    stress = read_csv_required(SCET_STRESS)
    summary = pd.concat([summary, read_csv_required(RAG_UTILITY_RESULTS)], ignore_index=True)
    stress = pd.concat([stress, read_csv_required(RAG_UTILITY_STRESS)], ignore_index=True)
    summary = summary[summary["retriever"].isin(EVIDENCE_LIST_METHODS | {"scet"})].copy()
    stress = stress[stress["retriever"].isin(EVIDENCE_LIST_METHODS | {"scet"})].copy()
    table = best_comparison_rows(summary, stress)

    x = np.arange(len(table))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25))

    ax = axes[0]
    comp_vals = table["comparison_precision"].to_numpy()
    scet_vals = table["scet_precision"].to_numpy()
    bars1 = ax.bar(x - width / 2, comp_vals, width, label="Best comparison", color=PALETTE["blue"], edgecolor=PALETTE["ink"], linewidth=0.45)
    bars2 = ax.bar(x + width / 2, scet_vals, width, label="SCET", color=PALETTE["orange"], edgecolor=PALETTE["ink"], linewidth=0.45)
    add_pair_labels(ax, list(bars1) + list(bars2), list(comp_vals) + list(scet_vals))
    ax.set_title("(a) Support precision")
    ax.set_ylabel("Precision")
    ax.set_xticks(x)
    ax.set_xticklabels(table["label"])
    ax.set_ylim(0, max(scet_vals.max(), comp_vals.max()) * 1.22)
    ax.legend(loc="upper left", ncol=1)
    finish_axis(ax)

    ax = axes[1]
    comp_vals = table["comparison_stress"].to_numpy()
    scet_vals = table["scet_stress"].to_numpy()
    bars1 = ax.bar(x - width / 2, comp_vals, width, label="Best comparison", color=PALETTE["green"], edgecolor=PALETTE["ink"], linewidth=0.45)
    bars2 = ax.bar(x + width / 2, scet_vals, width, label="SCET", color=PALETTE["purple"], edgecolor=PALETTE["ink"], linewidth=0.45)
    add_pair_labels(ax, list(bars1) + list(bars2), list(comp_vals) + list(scet_vals))
    ax.set_title("(b) Conflict-stress precision")
    ax.set_ylabel("Precision")
    ax.set_xticks(x)
    ax.set_xticklabels(table["label"])
    ax.set_ylim(0, max(scet_vals.max(), comp_vals.max()) * 1.24)
    finish_axis(ax)

    ax = axes[2]
    delta = table["recall_delta_vs_best"].to_numpy()
    colors = [PALETTE["orange"] if value >= -0.02 else "#9CA3AF" for value in delta]
    bars = ax.bar(x, delta, width=0.50, color=colors, edgecolor=PALETTE["ink"], linewidth=0.45)
    ax.axhline(0.0, color=PALETTE["ink"], linewidth=0.65)
    ax.axhline(-0.02, color=PALETTE["blue"], linewidth=0.75, linestyle="--")
    ax.text(2.45, -0.02, "gate", ha="right", va="bottom", fontsize=6.1, color=PALETTE["blue"])
    for bar, value in zip(bars, delta):
        va = "bottom" if value >= 0 else "top"
        offset = 0.0015 if value >= 0 else -0.0015
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.3f}", ha="center", va=va, fontsize=5.9)
    ax.set_title("(c) Recall@5 delta")
    ax.set_ylabel("SCET minus best")
    ax.set_xticks(x)
    ax.set_xticklabels(table["label"])
    ax.set_ylim(-0.026, 0.032)
    finish_axis(ax)

    fig.tight_layout(w_pad=1.25)
    outputs = save_bundle(fig, "ipm_scet_result_summary")
    plt.close(fig)
    return outputs


def main() -> None:
    outputs = make_mechanism_figure() + make_results_figure()
    for path in outputs:
        print(path.relative_to(PROJECT))


if __name__ == "__main__":
    main()
