from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "figures"


PALETTE = {
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#D55E00",
    "purple": "#CC79A7",
    "ink": "#1F2937",
    "grid": "#D1D5DB",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": 7.2,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.6,
            "axes.titlesize": 7.5,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def finish_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["ink"])
        spine.set_linewidth(0.7)
    ax.tick_params(direction="out", top=False, right=False, length=3.0, width=0.65, colors="black")
    ax.grid(True, axis="y", linewidth=0.35, alpha=0.55, color=PALETTE["grid"])


def save_bundle(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in (".pdf", ".svg", ".png", ".tif"):
        path = FIG_DIR / f"{stem}{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.025, "facecolor": "white"}
        if ext in {".png", ".tif"}:
            kwargs["dpi"] = 600
        if ext == ".tif":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def add_labels(
    ax: plt.Axes,
    bars,
    values: list[float],
    fmt: str = "{:.3f}",
    y_offsets: list[float] | None = None,
) -> None:
    ymax = max(max(values), 0.01)
    if y_offsets is None:
        y_offsets = [ymax * 0.025] * len(values)
    for bar, value, offset in zip(bars, values, y_offsets):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=5.8,
        )


def make_audit_stability_figure() -> list[Path]:
    set_style()
    datasets = ["SciFact", "FEVER", "HoVer"]
    x = np.arange(len(datasets))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25))

    ax = axes[0]
    scet = [0.4143, 0.3204, 0.8809]
    matched = [0.4143, 0.3199, 0.8761]
    bars_a = ax.bar(x - width / 2, matched, width, label="Matched view", color=PALETTE["blue"], edgecolor=PALETTE["ink"], linewidth=0.45)
    bars_b = ax.bar(x + width / 2, scet, width, label="SCET", color=PALETTE["orange"], edgecolor=PALETTE["ink"], linewidth=0.45)
    add_labels(ax, list(bars_a) + list(bars_b), matched + scet)
    ax.set_title("(a) Matched-volume precision")
    ax.set_ylabel("Support precision")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    finish_axis(ax)

    ax = axes[1]
    scet_stress = [0.1689, 0.0677, 0.4913]
    audit_stress = [0.2683, 0.0756, 0.5159]
    bars_a = ax.bar(x - width / 2, scet_stress, width, label="SCET", color=PALETTE["green"], edgecolor=PALETTE["ink"], linewidth=0.45)
    bars_b = ax.bar(x + width / 2, audit_stress, width, label="Audit", color=PALETTE["purple"], edgecolor=PALETTE["ink"], linewidth=0.45)
    add_labels(ax, list(bars_a) + list(bars_b), scet_stress + audit_stress)
    ax.set_title("(b) Conflict-pressure audit")
    ax.set_ylabel("Stress precision")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper left")
    finish_axis(ax)

    ax = axes[2]
    precision_mean = [0.3927, 0.3205, 0.8783]
    precision_sd = [0.0537, 0.0021, 0.0027]
    bars = ax.bar(x, precision_mean, width=0.48, yerr=precision_sd, capsize=3, color=PALETTE["orange"], edgecolor=PALETTE["ink"], linewidth=0.45)
    add_labels(ax, bars, precision_mean, y_offsets=[0.075, 0.022, 0.022])
    ax.set_title("(c) Calibration stability")
    ax.set_ylabel("Precision mean +/- sd")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    finish_axis(ax)

    fig.tight_layout(w_pad=1.2)
    outputs = save_bundle(fig, "figure3_scet_audit_stability")
    plt.close(fig)
    return outputs


def main() -> None:
    for path in make_audit_stability_figure():
        print(path.relative_to(PROJECT))


if __name__ == "__main__":
    main()
