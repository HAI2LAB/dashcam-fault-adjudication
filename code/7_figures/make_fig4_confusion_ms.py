#!/usr/bin/env python3
"""Supplementary Fig. 4: row-normalized confusion matrices.

Runtime dependencies: pandas, numpy, matplotlib (and Python's standard library).

Real CSV:
    python make_fig4_confusion.py --csv \
      ~/TS_code/result_e2e3_bb1/preds_internvl_32f_test_finetuning_bb1_eval.csv \
      --outdir ./figs

Smoke test with 30 SYNTHETIC rows (not paper results):
    python make_fig4_confusion.py --demo --outdir ./fig4_demo

Normal outputs: fig4_confusion.pdf and fig4_confusion.png.
Demo outputs have a _DEMO suffix and a visible synthetic-data notice.

Paper reference: supplementary.tex, Section S-III / fig:confusion.
Panels: situation, comparison, and five-band readout of 11-class predictions.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless SSH/server execution; no display required.
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

DEFAULT_CSV = (
    "~/TS_code/result_e2e3_bb1/"
    "preds_internvl_32f_test_finetuning_bb1_eval.csv"
)
FIGSIZE = (7.16, 2.6)
EXPECTED_N = 2837
PNG_DPI = 150
DEMO_SEED = 20260911

SITUATION_NAMES = (
    "Go straight", "Left turn", "Right turn", "U-turn", "Reverse",
    "Lane change/Merge", "Overtaking", "Rear-end/Collision", "Roundabout",
    "Centerline violation", "Stop/Park", "Pedestrian/Crossing", "Other",
)
# Fixed display order, matching the paper's per-situation table.
# Reorder BOTH rows and columns; never sort by observed CSV frequency.
SITUATION_ORDER = (0, 7, 11, 5, 1, 4, 6, 10, 12, 2, 8, 3, 9)
COMPARISON_NAMES = ("B>A", "A≈B", "A>B")
BAND_NAMES = ("0–10", "20–30", "40–60", "70–80", "90–100")
# Input ratio class: 0,10,20,30,40,50,60,70,80,90,100.
def ratio_to_band(values: np.ndarray) -> np.ndarray:
    """Identical to e2e_model3.ratio_to_class_5: <=10->0, <=30->1, <=60->2, <=80->3, else 4."""
    v = np.asarray(values)
    return np.select([v <= 10, v <= 30, v <= 60, v <= 80], [0, 1, 2, 3], default=4).astype(np.int64)
TITLES = (
    "(a) Situation (13-way)",
    "(b) Comparison (3-way)",
    "(c) Fault band (5-way)",
)
PAPER_ACC = (0.610, 0.671, 0.509)
REQUIRED = (
    "video", "sit_gt", "sit_pred", "cmp_gt", "cmp_pred", "ratio_gt",
    "ratio_pred", "ratio_abs_err", "within10",
)
RC = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.usetex": False, "axes.grid": False,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.transparent": True, "pdf.fonttype": 42, "ps.fonttype": 42,
}


def expanded_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def integer_column(df: pd.DataFrame, column: str, allowed: np.ndarray) -> np.ndarray:
    """Reject NaNs, nonintegers and out-of-schema values; never drop/round rows."""
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isin(values, allowed)
    if not valid.all():
        bad = np.flatnonzero(~valid)[:5]
        examples = ", ".join(
            f"CSV line {i + 2}: {df[column].iloc[i]!r}" for i in bad
        )
        raise ValueError(
            f"Invalid {column}; expected {allowed.tolist()}. {examples}. "
            "No rows were dropped and no labels were rounded."
        )
    return values.astype(np.int64)


def load_data(path: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    if not path.is_file():
        raise ValueError(f"CSV file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [name for name in REQUIRED if name not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    if df.empty:
        raise ValueError("The input CSV has no data rows.")
    result = {}
    for prefix, allowed in (
        ("sit", np.arange(13)), ("cmp", np.arange(3)),
        ("ratio", np.arange(0, 101)),   # raw labels may be non-multiples of 10
    ):
        for suffix in ("gt", "pred"):
            key = f"{prefix}_{suffix}"
            result[key] = integer_column(df, key, allowed)
    # These stored diagnostics are not used to construct the confusion matrices.
    errors = np.abs(result["ratio_gt"] - result["ratio_pred"])
    recorded = pd.to_numeric(df["ratio_abs_err"], errors="coerce").to_numpy()
    if not np.allclose(recorded, errors, equal_nan=False):
        print("WARNING: ratio_abs_err differs from the raw ratio columns.", file=sys.stderr)
    recorded_within = df["within10"].astype(str).str.strip().str.lower().map(
        {"0": 0, "0.0": 0, "false": 0, "1": 1, "1.0": 1, "true": 1}
    ).to_numpy(dtype=float)
    if not np.array_equal(recorded_within, (errors <= 10).astype(float)):
        print("WARNING: within10 differs from abs(ratio_gt-ratio_pred) <= 10.", file=sys.stderr)
    if df["video"].duplicated().any():
        print("WARNING: duplicate video identifiers; every CSV row is counted.", file=sys.stderr)
    return df, result


def confusion_counts(gt: np.ndarray, pred: np.ndarray, classes: int) -> np.ndarray:
    counts = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(counts, (gt, pred), 1)
    if int(counts.sum()) != len(gt):
        raise RuntimeError("Confusion-matrix total does not match the input length.")
    return counts


def prepare_panels(data: Dict[str, np.ndarray]) -> List[Tuple[np.ndarray, Tuple[str, ...]]]:
    situation = confusion_counts(data["sit_gt"], data["sit_pred"], 13)
    order = np.array(SITUATION_ORDER)
    situation = situation[np.ix_(order, order)]
    comparison = confusion_counts(data["cmp_gt"], data["cmp_pred"], 3)
    # Fold GT and predictions identically; do NOT retrain or reinterpret cmp labels.
    band_gt = ratio_to_band(data["ratio_gt"])
    band_pred = ratio_to_band(data["ratio_pred"])
    bands = confusion_counts(band_gt, band_pred, 5)
    return [
        (situation, tuple(SITUATION_NAMES[i] for i in SITUATION_ORDER)),
        (comparison, COMPARISON_NAMES), (bands, BAND_NAMES),
    ]


def row_normalize(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    support = counts.sum(axis=1)
    normalized = np.divide(
        counts, support[:, None], out=np.zeros_like(counts, dtype=float),
        where=support[:, None] > 0,
    )
    if not np.allclose(normalized[support > 0].sum(axis=1), 1.0):
        raise RuntimeError("A supported confusion-matrix row does not sum to 1.")
    # n=0 rows are undefined, displayed as all-zero blank rows, explicitly labeled n=0.
    return normalized, support


class FigureLayer(Artist):
    """Draw an independent single-chart Figure as original vectors on the output page.

    This uses Matplotlib's renderer directly: no PNG embedding, PDF conversion,
    PyMuPDF, Pillow calls, external fonts, or additional runtime dependencies.
    """
    def __init__(self, source: Figure):
        super().__init__()
        self.source = source

    def _sync(self) -> None:
        self.source.set_dpi(self.figure.dpi)
        self.source.transFigure.set_matrix(self.figure.transFigure.get_matrix().copy())

    def draw(self, renderer) -> None:
        self._sync()
        self.source.draw(renderer)
        self.stale = False

    def get_tightbbox(self, renderer=None):
        self._sync()
        bbox = self.source.get_tightbbox(renderer)
        return None if bbox is None else bbox.transformed(self.source.dpi_scale_trans)


def independent_figure() -> Figure:
    fig = Figure(figsize=FIGSIZE, dpi=PNG_DPI, facecolor="none")
    # A mutable affine transform permits vector-safe composition at PDF/PNG DPI.
    fig.transFigure = Affine2D(fig.transFigure.get_matrix().copy())
    fig.transSubfigure = fig.transFigure
    fig.patch.set_visible(False)
    return fig


def layout(page: Figure, panels, n: int):
    """Reserve label space; matrix widths themselves are exactly 13:3:5."""
    renderer = page.canvas.get_renderer()
    prop = FontProperties(family="DejaVu Sans", size=7)
    def text_width(text: str) -> float:
        return renderer.get_text_width_height_descent(text, prop, False)[0] / page.dpi
    # Reserve four-digit supports even for the tiny demo, for consistent geometry.
    digits = "8" * max(4, len(str(n)))
    label_width = [max(text_width(f"{label} (n={digits})") for label in labels)
                   for _, labels in panels]
    width, height = FIGSIZE
    left = label_width[0] + 0.24  # Ground truth title, tick padding and outer margin.
    gap_b, gap_c = label_width[1] + 0.075, label_width[2] + 0.075
    bar_gap, bar_width, right = 0.10, 0.075, 0.235
    available = width - left - gap_b - gap_c - bar_gap - bar_width - right
    if available <= 2.5:
        raise ValueError("Labels do not fit: increase FIGSIZE[0] at the script top.")
    cell_width = available / 21.0
    # Blank GridSpec columns reserve label gutters and a single shared colorbar.
    ratios = [13, gap_b / cell_width, 3, gap_c / cell_width, 5,
              bar_gap / cell_width, bar_width / cell_width]
    grid = GridSpec(
        1, 7, figure=page, width_ratios=ratios, wspace=0,
        left=left / width, right=(width - right) / width,
        bottom=0.98 / height, top=2.40 / height,
    )
    boxes = [grid[0, i].get_position(page) for i in (0, 2, 4)]
    bar_box = grid[0, 6].get_position(page)
    # Center over the decorated panel (not just its narrow matrix), avoiding title collisions.
    title_x = [
        (boxes[0].x0 + boxes[0].x1) / 2,
        boxes[1].x0 + boxes[1].width / 2 - 0.35 / width,
        boxes[2].x0 + boxes[2].width / 2 - 0.23 / width,
    ]
    return boxes, bar_box, title_x


def make_panel(counts, labels, index, box, title_x, norm):
    source = independent_figure()
    ax = source.add_axes(box.bounds)
    normalized, support = row_normalize(counts)
    size = len(labels)
    edges = np.arange(size + 1)
    ax.pcolormesh(
        edges, edges, normalized, cmap="Blues", norm=norm, shading="flat",
        edgecolors="white", linewidth=0.5, antialiased=False, rasterized=False,
    )
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)  # GT row zero is at the top.
    ax.set_aspect("auto")
    centers = np.arange(size) + 0.5
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
    ax.set_yticks(centers)
    ax.set_yticklabels(
        [f"{label} (n={int(k)})" for label, k in zip(labels, support)],
        rotation=0, fontsize=7,
    )
    ax.tick_params(axis="both", which="both", length=0, pad=2)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if index == 0:
        ax.set_xlabel("Predicted", fontsize=7, labelpad=3)
        ax.set_ylabel("Ground truth", fontsize=7, labelpad=3)
    # STIXGeneral is bundled with Matplotlib; compact numerals avoid cell collisions
    # at 6 pt without reducing font size or requiring a server-installed font.
    for row, col in np.argwhere(normalized >= 0.05):
        value = float(normalized[row, col])
        ax.text(
            col + 0.5, row + 0.5, f"{value:.2f}", ha="center", va="center",
            fontsize=6, fontfamily="STIXGeneral",
            color="white" if value > 0.5 else "black", clip_on=True,
        )
    source.text(title_x, 2.49 / FIGSIZE[1], TITLES[index],
                ha="center", va="center", fontsize=8)
    return source


def make_page(panels, n: int, demo: bool) -> Figure:
    page = Figure(figsize=FIGSIZE, dpi=PNG_DPI, facecolor="none")
    FigureCanvasAgg(page)
    boxes, bar_box, title_x = layout(page, panels, n)
    norm = Normalize(vmin=0.0, vmax=1.0)
    for index, ((counts, labels), box, tx) in enumerate(zip(panels, boxes, title_x)):
        page.add_artist(FigureLayer(make_panel(counts, labels, index, box, tx, norm)))
    # The colorbar is a separate legend layer using the same fixed normalization.
    legend = independent_figure()
    cax = legend.add_axes(bar_box.bounds)
    colorbar = ColorbarBase(cax, cmap=matplotlib.cm.Blues, norm=norm,
                           orientation="vertical", ticks=np.linspace(0, 1, 6))
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=7, length=2, width=0.5, pad=2)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)  # Colorbars otherwise may rasterize.
        colorbar.solids.set_edgecolor("face")  # Avoid white seams in PDF viewers.
        colorbar.solids.set_antialiased(False)
    page.add_artist(FigureLayer(legend))
    if demo:
        page.text(0.95, 0.10, "SYNTHETIC DATA · 30 rows\nNot paper results",
                  ha="right", va="center", fontsize=7, color="#666666")
    # Keep the full nominal page when using bbox_inches='tight', without shrinking text.
    page.add_artist(Rectangle(
        (0, 0), 1, 1, transform=page.transFigure, facecolor="none", edgecolor="none",
        clip_on=False,
    ))
    page.canvas.draw()
    return page


def print_accuracy(panels, n: int, demo: bool) -> None:
    print(f"Rows: {n:,}; matrix convention: rows=GT, columns=Predicted")
    print("Accuracy = trace(raw count matrix) / N; NOT trace(row-normalized matrix).")
    if demo:
        print("SYNTHETIC 30-row smoke test: these scores are NOT paper results.")
    elif n != EXPECTED_N:
        print(f"WARNING: expected {EXPECTED_N:,} test rows, received {n:,}.")
    for title, (counts, labels), target in zip(TITLES, panels, PAPER_ACC):
        correct = int(np.trace(counts))
        accuracy = correct / n
        status = "demo only" if demo else ("MATCH (3 d.p.)" if f"{accuracy:.3f}" == f"{target:.3f}" else "DIFF")
        print(f"{title}: {correct}/{n} = {accuracy:.6f}; "
              f"paper={target:.3f}; delta={accuracy-target:+.6f} [{status}]")
        zero = [labels[i] for i in np.flatnonzero(counts.sum(axis=1) == 0)]
        if zero:
            print("  n=0 / blank rows: " + ", ".join(zero))


def create_demo(path: Path, seed: int) -> None:
    """Pure random data, not fitted to any manuscript result or claimed error pattern."""
    rng = np.random.default_rng(seed)
    n = 30
    df = pd.DataFrame({
        "video": [f"FAKE_SMOKE_TEST_{i:03d}.mp4" for i in range(n)],
        "sit_gt": rng.integers(0, 13, n), "sit_pred": rng.integers(0, 13, n),
        "cmp_gt": rng.integers(0, 3, n), "cmp_pred": rng.integers(0, 3, n),
        "ratio_gt": rng.integers(0, 11, n) * 10,
        "ratio_pred": rng.integers(0, 11, n) * 10,
    })
    df["ratio_abs_err"] = (df["ratio_gt"] - df["ratio_pred"]).abs()
    df["within10"] = (df["ratio_abs_err"] <= 10).astype(int)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--csv", type=str, help=f"Prediction CSV; default: {DEFAULT_CSV}")
    group.add_argument("--demo", action="store_true", help="Generate/run 30 synthetic rows; never use for the paper.")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory (default: current directory).")
    parser.add_argument("--seed", type=int, default=DEMO_SEED, help="Random seed used only by --demo.")
    args = parser.parse_args()
    try:
        outdir = expanded_path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        if args.demo:
            csv_path = outdir / "fig4_confusion_demo_input_30.csv"
            create_demo(csv_path, args.seed)
        else:
            csv_path = expanded_path(args.csv or DEFAULT_CSV)
        df, data = load_data(csv_path)
        panels = prepare_panels(data)
        print(f"Input: {csv_path}")
        print_accuracy(panels, len(df), args.demo)
        stem = "fig4_confusion_DEMO" if args.demo else "fig4_confusion"
        with matplotlib.rc_context(RC):
            page = make_page(panels, len(df), args.demo)
            page.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0,
                         transparent=True, metadata={
                             "Title": "SYNTHETIC TEST ONLY" if args.demo else "Supplementary Fig. 4: Confusion matrices",
                             "Subject": "Row-normalized; situation / comparison / five-band fault ratio",
                         })
            page.savefig(outdir / f"{stem}.png", bbox_inches="tight", pad_inches=0,
                         transparent=True, dpi=PNG_DPI)
            page.clear()
        print(f"Saved: {outdir / (stem + '.pdf')}")
        print(f"Saved: {outdir / (stem + '.png')} (150 dpi, transparent)")
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
