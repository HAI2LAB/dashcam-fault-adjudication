#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make Supplementary Fig. 3 from pre-extracted frames (1-column or 2-column).

Python 3.8+; third-party dependencies: pandas, numpy, matplotlib, Pillow.
No video decoding, impact detection, network access, or extra anonymization.

Typical execution:
    python make_fig3_qualitative.py --outdir ./figs --gamma 1.2
    python make_fig3_qualitative.py --clips fig3_clips.txt --outdir ./figs
    python make_fig3_qualitative.py --columns 2 --frames 4 --outdir ./figs_grid
    python make_fig3_qualitative.py --demo --columns 2 --frames 4 --outdir ./fig3_grid_demo
    python make_fig3_qualitative.py --demo --outdir ./fig3_demo
    python make_fig3_qualitative.py --write-default-clips fig3_clips.txt

Default paths below can be overridden with --csv, --qwen-csv, --frame-roots.
If --clips is omitted, a missing ./fig3_clips.txt is initialized with the exact
14-clip selection below. Existing configuration files are NEVER overwritten.

Semantics:
  * Situation and comparison ticks mean exact index equality.
  * GT ratio is half-up quantized to the legal 10-point target for display.
    A ratio tick means prediction == that displayed target (NOT within10).
    Predictions are NEVER rounded. Raw GT and the display conversion are logged.
  * Gamma > 1 brightens: output = 255 * (input / 255) ** (1 / gamma).
  * The first EXISTING clip folder in root priority order is used. If incomplete
    or unreadable for the selected --frames, the clip is skipped; roots are not mixed.
  * Block-condition disagreements are warnings; selections are not reclassified.
  * Defaults remain --columns 1 --frames 6. --frames 4 selects exactly
    t-1.0s, t-0.5s, t+0.0s, t+0.5s (not t-1.5s or t+1.0s).
  * Two-column mode fixes blocks to (a)|(b), then (c)|(d); a missing block
    never shifts another block into its slot. Each column is 3.5 in wide
    including internal padding; the inter-column gutter is 0.16 in.
  * Two-column captions have exactly two lines at 7 pt (6.75/6.5 pt only
    if needed, measured using actual font metrics). No label is abbreviated.
    Timestamps appear once per column above its first rendered strip.
  * Height is automatic: balanced 8-clip, 4-frame input is about 4.4 in
    for 16:9 frames (about 4.6 in including the synthetic-demo banner).
    Height depends on max(n_a,n_b) + max(n_c,n_d), not just total clips.
  * The 9.2-inch cap reduces PHOTO height, never annotation font sizes.
    Aspect ratios and 2-pt gaps are preserved; narrower strips are centered.
    Gamma defaults to 1.0; no new automatic image post-processing is added.

Source note: the attached supplementary.tex caption/S-I contains blocks (a)-(c)
only. Block (d) follows the additional user specification and requires a caption
update. Recovery examples are illustrative, not a causal test of soft chaining.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless remote-server execution; no GUI required.
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextToPath
from PIL import Image, ImageDraw, ImageFont, ImageOps

# --------------------------- USER CONFIGURATION ---------------------------
DEFAULT_CSV = "~/TS_code/result_e2e3_bb1/preds_internvl_32f_test_finetuning_bb1_eval.csv"
DEFAULT_QWEN_CSV = "~/TS_code/result_e2e3_bb1/preds_qwen3_32f_test_finetuning_bb1_eval.csv"
DEFAULT_ROOTS = (
    "~/TS_code/impact_shots_ms_0908_manual/",
    "~/TS_code/impact_shots_ms_0907/",
    "~/TS_code/backup/impact_shots/",
)
DEFAULT_CLIPS = """# (a) Successful cases
a bb_1_200329_vehicle_125_171
a bb_1_190105_bike_121_068
a bb_1_210701_two-wheeled-vehicle_247_22003
a bb_1_150621_vehicle_218_34287
a bb_1_180610_vehicle_251_21646
a bb_1_160728_two-wheeled-vehicle_118_005
# (b) Situation-error tolerance
b bb_1_220116_two-wheeled-vehicle_47_004
b bb_1_190727_bike_259_20146
b bb_1_130419_vehicle_204_51863
# (c) Characteristic failures
c bb_1_200220_vehicle_47_250
c bb_1_140326_vehicle_253_21570
c bb_1_181123_vehicle_237_21720
# (d) Backbone divergence (InternVL3 correct, Qwen3 wrong)
d bb_1_220402_vehicle_233_31959
d bb_1_200104_bike_230_22399
"""
SITUATIONS = (
    "Go straight", "Left turn", "Right turn", "U-turn", "Reverse",
    "Lane change/Merge", "Overtaking", "Rear-end/Collision", "Roundabout",
    "Centerline violation", "Stop/Park", "Pedestrian/Crossing", "Other",
)
COMPARISONS = ("B>A", "A≈B", "A>B")
BLOCKS = {
    "a": "(a) Successful cases",
    "b": "(b) Situation-error tolerance: situation wrong, ratio correct",
    "c": "(c) Characteristic failures",
    "d": "(d) Backbone divergence",
}
FRAME_NAMES = (
    "t-1.5s.jpg", "t-1.0s.jpg", "t-0.5s.jpg",
    "t+0.0s.jpg", "t+0.5s.jpg", "t+1.0s.jpg",
)
TIME_LABELS = ("t−1.5s", "t−1.0s", "t−0.5s", "t+0.0s", "t+0.5s", "t+1.0s")
FOUR_FRAME_INDICES = (1, 2, 3, 4)
WIDTH_IN, DPI = 7.16, 300
GREEN, RED, ORANGE, BLACK = "#2ca02c", "#d62728", "#ff7f0e", "#111111"
# All geometric constants below are physical PDF points, NOT pixels.
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 8.0, 7.0, 7.0
FRAME_GAP, HEADER_H, BLOCK_GAP, TIME_H = 2.0, 12.0, 5.0, 8.0
ANNOTATION_GAP, ANNOTATION_LINE_H, ROW_GAP = 2.0, 9.0, 3.0
FONT_ANNOTATION, FONT_TITLE, FONT_TIME = 7.0, 8.0, 6.0
# Column width includes a 2.5-pt inset on each side to protect the border.
GRID_COLUMN_W = 3.5 * 72
GRID_GUTTER = WIDTH_IN * 72 - 2 * GRID_COLUMN_W
GRID_PAD, GRID_ROW_GAP = 2.5, 8.0
# Previously the gap after a block was an 8-pt final row gap + 12-pt band gap.
# Reserve row gaps ONLY between strips and halve that actual 20-pt gap to 10 pt.
GRID_BAND_GAP = (8.0 + 12.0) / 2.0
GRID_BOTTOM_PAD = 1.0
LETTERBOX_MIN_ROWS = 2
LETTERBOX_MAX_FRACTION = 0.25  # Reject excessive edge runs, rather than cut a scene.
GRID_ANNOTATION_SIZES = (7.0, 6.75, 6.5)
GRID_PAIRS = (("a", "b"), ("c", "d"))
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


class SkipClip(ValueError):
    """An individual unusable clip; warn and continue with the remaining clips."""


def warn(message: str) -> None:
    print("WARNING: " + message, file=sys.stderr, flush=True)


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def normalized_id(value: object) -> str:
    name = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    return name[:-4] if name.lower().endswith(".mp4") else name


def quantize_gt(value: int) -> int:
    """Exact half-up rounding for nonnegative integer percentages: 25 -> 30."""
    if not 0 <= value <= 100:
        raise ValueError("Ratio GT must be in [0, 100].")
    return int(math.floor(value / 10.0 + 0.5) * 10)


def ratio_text(value: int) -> str:
    return "{}:{}".format(value, 100 - value)


@dataclass(frozen=True)
class Selection:
    block: str
    clip_id: str


@dataclass(frozen=True)
class Prediction:
    sit_gt: int
    sit_pred: int
    cmp_gt: int
    cmp_pred: int
    ratio_gt: int
    ratio_pred: int


@dataclass(frozen=True)
class Run:
    text: str
    color: str = BLACK
    weight: str = "normal"


@dataclass
class Clip:
    selection: Selection
    folder: Path
    images: List[Image.Image]
    aspects: List[float]
    iv: Prediction
    qwen: Optional[Prediction]
    groups: List[List[Run]]
    lines: List[List[Run]]
    condition_ok: bool
    annotation_size: float = FONT_ANNOTATION
    letterbox_top_px: int = 0
    letterbox_bottom_px: int = 0
    letterbox_status: str = "disabled"


@dataclass(frozen=True)
class Layout:
    width_pt: float
    height_pt: float
    image_h: float
    natural_height_pt: float
    n_blocks: int
    banner_h: float


@dataclass(frozen=True)
class GridBand:
    blocks: Tuple[str, str]
    title_lines: Tuple[Tuple[str, ...], Tuple[str, ...]]
    header_h: float
    timestamps: Tuple[bool, bool]
    time_h: float
    row_count: int


@dataclass(frozen=True)
class GridLayout:
    width_pt: float
    height_pt: float
    image_h: float
    natural_height_pt: float
    banner_h: float
    bands: Tuple[GridBand, ...]


def selected_frames(count: int = 6) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if count not in (4, 6):
        raise ValueError("--frames must be 4 or 6")
    indices = FOUR_FRAME_INDICES if count == 4 else range(6)
    return tuple(FRAME_NAMES[i] for i in indices), tuple(TIME_LABELS[i] for i in indices)


def read_selection(path: Path) -> List[Selection]:
    selected: List[Selection] = []
    seen = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2 or parts[0] not in BLOCKS:
            warn("{}:{}: expected 'a|b|c|d clip_id'; skipping line.".format(path, lineno))
            continue
        block, clip_id = parts[0], normalized_id(parts[1])
        if any(s in parts[1] for s in ("/", "\\")) or clip_id in ("", ".", ".."):
            warn("{}:{}: clip_id must be a folder basename; skipping.".format(path, lineno))
            continue
        key = (block, clip_id)
        if key in seen:
            warn("Duplicate selection {} {}; keeping its first occurrence.".format(*key))
            continue
        seen.add(key)
        selected.append(Selection(block, clip_id))
    # Always (a), (b), (c), (d), preserving file order WITHIN each block.
    return [item for block in BLOCKS for item in selected if item.block == block]


def read_predictions(path: Path, model: str) -> Dict[str, List[dict]]:
    """Index rows without silently taking the first of duplicate clip IDs."""
    required = {"video", "sit_gt", "sit_pred", "cmp_gt", "cmp_pred", "ratio_gt", "ratio_pred"}
    try:
        data = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        warn("{} CSV unavailable: {} ({})".format(model, path, exc))
        return {}
    missing = required - set(data.columns)
    if missing:
        warn("{} CSV missing columns {}; its clips will be skipped.".format(model, sorted(missing)))
        return {}
    result: Dict[str, List[dict]] = {}
    for row in data.to_dict(orient="records"):
        if pd.isna(row["video"]):
            continue
        result.setdefault(normalized_id(row["video"]), []).append(row)
    return result


def get_prediction(index: Dict[str, List[dict]], clip_id: str, model: str) -> Prediction:
    rows = index.get(clip_id, [])
    if not rows:
        raise SkipClip("{} CSV row not found".format(model))
    if len(rows) != 1:
        raise SkipClip("{} has {} rows for this clip; ambiguous match".format(model, len(rows)))
    vals = {}
    for column, high in (("sit_gt", 12), ("sit_pred", 12), ("cmp_gt", 2),
                         ("cmp_pred", 2), ("ratio_gt", 100), ("ratio_pred", 100)):
        try:
            value = float(rows[0][column])
        except (TypeError, ValueError):
            raise SkipClip("{} {} is not numeric".format(model, column))
        if not math.isfinite(value) or not value.is_integer() or not 0 <= value <= high:
            raise SkipClip("{} {}={} must be an integer in [0, {}]".format(model, column, value, high))
        vals[column] = int(value)
    return Prediction(**vals)


def gamma_lut(gamma: float) -> List[int]:
    return np.rint(255.0 * (np.arange(256, dtype=float) / 255.0) ** (1.0 / gamma)).astype(np.uint8).tolist()


def crop_common_black_rows(images: Sequence[Image.Image]
                           ) -> Tuple[List[Image.Image], int, int, str]:
    """Conservative crop: remove only exact-black horizontal edge rows.

    All frames must have the same dimensions. A row is a border candidate only
    when EVERY pixel in EVERY RGB channel is zero. The intersection of the
    edge runs across all selected frames is used, so the viewport is identical
    throughout a clip. JPEG near-black noise and dark scene content are NOT
    removed using a brightness threshold. This is a geometric pixel test, not
    a semantic claim that every dark region in a video is a letterbox.
    """
    if not images:
        return [], 0, 0, "empty input"
    sizes = {image.size for image in images}
    if len(sizes) != 1:
        return list(images), 0, 0, "kept: inconsistent source dimensions"
    width, height = images[0].size
    runs = []
    for image in images:
        rgb = np.asarray(image.convert("RGB"))
        all_black = np.all(rgb == 0, axis=(1, 2))
        nonblack = np.flatnonzero(~all_black)
        if nonblack.size == 0:
            return list(images), 0, 0, "kept: entirely black frame / ambiguous"
        runs.append((int(nonblack[0]), int(height - 1 - nonblack[-1])))
    top = min(t for t, _ in runs)
    bottom = min(b for _, b in runs)
    top = top if top >= LETTERBOX_MIN_ROWS else 0
    bottom = bottom if bottom >= LETTERBOX_MIN_ROWS else 0
    if max(top, bottom) > height * LETTERBOX_MAX_FRACTION:
        return list(images), 0, 0, "kept: excessive black region / ambiguous"
    if top + bottom >= height // 2:
        return list(images), 0, 0, "kept: insufficient nonblack image height"
    if not (top or bottom):
        return list(images), 0, 0, "kept: no common full-width exact-black edge rows"
    # Assertion guarantees that no nonzero RGB sample is discarded.
    for image in images:
        rgb = np.asarray(image.convert("RGB"))
        assert not top or np.all(rgb[:top] == 0)
        assert not bottom or np.all(rgb[height - bottom:] == 0)
    cropped = [image.crop((0, top, width, height - bottom)) for image in images]
    return cropped, top, bottom, "cropped: common exact-black edge rows only"


def load_frames(clip_id: str, roots: Sequence[Path], gamma: float,
                row_height: float, frame_names: Sequence[str] = FRAME_NAMES,
                crop_letterbox: bool = False, crop_audit: Optional[dict] = None
                ) -> Tuple[Path, List[Image.Image], List[float]]:
    folder = next((root / clip_id for root in roots if (root / clip_id).is_dir()), None)
    if folder is None:
        raise SkipClip("frame folder not found in any configured root")
    missing = [name for name in frame_names if not (folder / name).is_file()]
    if missing:
        raise SkipClip("first-priority folder {} is incomplete: {}; not mixing/falling back".format(folder, ", ".join(missing)))
    images = []
    for name in frame_names:
        try:
            with Image.open(folder / name) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.load()
            images.append(image)
        except (OSError, ValueError, SyntaxError) as exc:
            raise SkipClip("cannot read {}/{}: {}".format(folder, name, exc))
    if crop_letterbox:
        images, top, bottom, note = crop_common_black_rows(images)
    else:
        top, bottom, note = 0, 0, "disabled (original single-column behavior or explicit opt-out)"
    if crop_audit is not None:
        crop_audit.update(top_px=top, bottom_px=bottom, status=note)
    # Cropping precedes gamma and resizing, so the detector sees ORIGINAL pixels.
    aspects, output = [], []
    lut = gamma_lut(gamma) * 3
    max_h_px = max(1, math.ceil(row_height * DPI))
    for image in images:
        w, h = image.size
        aspects.append(w / h)
        if h > max_h_px:
            image = image.resize((max(1, round(w * max_h_px / h)), max_h_px), LANCZOS)
        if gamma != 1.0:
            image = image.point(lut)
        output.append(image)
    return folder, output, aspects


def pred_run(text: str, ok: bool) -> Run:
    """Prediction text colored by correctness: green regular = correct, red bold = wrong."""
    return Run(text, GREEN if ok else RED, "normal" if ok else "bold")


def make_groups(block: str, iv: Prediction, qwen: Optional[Prediction]) -> Tuple[List[List[Run]], bool]:
    gt = quantize_gt(iv.ratio_gt)
    sit_ok, cmp_ok, ratio_ok = iv.sit_gt == iv.sit_pred, iv.cmp_gt == iv.cmp_pred, gt == iv.ratio_pred
    if block == "d":
        if qwen is None:
            raise SkipClip("Qwen3-VL prediction required for block (d)")
        qwen_ok = gt == qwen.ratio_pred
        groups = [
            [Run("Sit: GT " + SITUATIONS[iv.sit_gt])],
            [Run("Ratio: GT {} / InternVL3 ".format(ratio_text(gt))), pred_run(ratio_text(iv.ratio_pred), ratio_ok),
             Run(" / Qwen3-VL "), pred_run(ratio_text(qwen.ratio_pred), qwen_ok)],
        ]
        return groups, ratio_ok and not qwen_ok
    groups = [
        [Run("Sit: GT {} / Pred ".format(SITUATIONS[iv.sit_gt])), pred_run(SITUATIONS[iv.sit_pred], sit_ok)],
        [Run("Cmp: GT {} / Pred ".format(COMPARISONS[iv.cmp_gt])), pred_run(COMPARISONS[iv.cmp_pred], cmp_ok)],
        [Run("Ratio: GT {} / Pred ".format(ratio_text(gt))), pred_run(ratio_text(iv.ratio_pred), ratio_ok)],
    ]
    condition = {"a": sit_ok and cmp_ok and ratio_ok,
                 "b": not sit_ok and ratio_ok,
                 "c": not (sit_ok and cmp_ok and ratio_ok)}[block]
    return groups, condition


def joined_text(groups: Sequence[Sequence[Run]]) -> str:
    return "   ".join("".join(run.text for run in group) for group in groups)


def text_width(text: str, size: float = FONT_ANNOTATION, weight: str = "normal") -> float:
    prop = FontProperties(family="DejaVu Sans", size=size, weight=weight)
    return TextToPath().get_text_width_height_descent(text, prop, ismath=False)[0]


def annotation_lines(groups: List[List[Run]], available: float) -> List[List[Run]]:
    """Known class names fit one line at 7 pt; wrap only at a head boundary."""
    lines: List[List[Run]] = [[]]
    current = 0.0
    for group in groups:
        width = sum(text_width(run.text, weight=run.weight) for run in group)
        if width > available:
            raise ValueError("One annotation group exceeds figure width at 7 pt; edit layout, not text content.")
        gap = text_width("   ") if lines[-1] else 0.0
        if lines[-1] and current + gap + width > available:
            lines.append([])
            current, gap = 0.0, 0.0
        if gap:
            lines[-1].append(Run("   "))
        lines[-1].extend(group)
        current += gap + width
    return lines


def grid_annotation_lines(groups: List[List[Run]], block: str,
                          available: float) -> Tuple[List[List[Run]], float]:
    """Keep the requested semantic two-line grouping; never shrink below 6.5 pt."""
    if block == "d":
        lines = [list(groups[0]), list(groups[1])]
    else:
        lines = [list(groups[0]), list(groups[1]) + [Run("   ")] + list(groups[2])]
    for size in GRID_ANNOTATION_SIZES:
        longest = max(sum(text_width(run.text, size, run.weight) for run in line) for line in lines)
        if longest <= available - 1.0:  # One-point reserve for renderer differences.
            return lines, size
    raise ValueError("Two-line annotation does not fit a 3.5-in column at 6.5 pt: "
                     + joined_text(lines) + ". Use --columns 1; content was not shortened.")


def grid_title_lines(block: str) -> Tuple[str, ...]:
    # One identical-height title line per panel; the (b) explanation is in caption.
    title = "(b) Situation-error tolerance" if block == "b" else BLOCKS[block]
    if text_width(title, FONT_TITLE, "bold") > GRID_COLUMN_W - 2 * GRID_PAD:
        raise ValueError("Block title does not fit a grid column: " + title)
    return (title,)


def collect_clips(selected: Sequence[Selection], iv_index: Dict[str, List[dict]],
                  qwen_index: Dict[str, List[dict]], roots: Sequence[Path],
                  gamma: float, row_height: float, columns: int = 1,
                  frames: int = 6, crop_letterbox: bool = True) -> List[Clip]:
    clips = []
    frame_names, _ = selected_frames(frames)
    for selection in selected:
        try:
            iv = get_prediction(iv_index, selection.clip_id, "InternVL3")
            qwen = get_prediction(qwen_index, selection.clip_id, "Qwen3-VL") if selection.block == "d" else None
            if qwen is not None and (iv.sit_gt, iv.cmp_gt, iv.ratio_gt) != (qwen.sit_gt, qwen.cmp_gt, qwen.ratio_gt):
                raise SkipClip("GT disagreement between the two CSVs; cannot compare backbones reliably")
            crop_audit = {}
            folder, images, aspects = load_frames(
                selection.clip_id, roots, gamma, row_height, frame_names,
                crop_letterbox=(columns == 2 and crop_letterbox), crop_audit=crop_audit)
            groups, condition = make_groups(selection.block, iv, qwen)
            if columns == 2:
                lines, size = grid_annotation_lines(groups, selection.block, GRID_COLUMN_W - 2 * GRID_PAD)
            else:
                lines = annotation_lines(groups, WIDTH_IN * 72 - 2 * MARGIN_X)
                size = FONT_ANNOTATION
            clip = Clip(selection, folder, images, aspects, iv, qwen, groups, lines, condition, size)
            clip.letterbox_top_px = crop_audit.get("top_px", 0)
            clip.letterbox_bottom_px = crop_audit.get("bottom_px", 0)
            clip.letterbox_status = crop_audit.get("status", "disabled")
            clips.append(clip)
            if columns == 2:
                print("  Letterbox {}: top={} px, bottom={} px; {}".format(
                    selection.clip_id, clip.letterbox_top_px,
                    clip.letterbox_bottom_px, clip.letterbox_status), flush=True)
            print("[{}] {}\n  frames: {}\n  {}".format(selection.block, selection.clip_id, folder, joined_text(groups)), flush=True)
            if iv.ratio_gt != quantize_gt(iv.ratio_gt):
                print("  GT display/target half-up: {} -> {}; ticks compare against {} (prediction unrounded).".format(
                    iv.ratio_gt, quantize_gt(iv.ratio_gt), quantize_gt(iv.ratio_gt)), flush=True)
            if not condition:
                warn("{}: does not meet block ({}) outcome condition; retaining the explicitly selected clip.".format(selection.clip_id, selection.block))
            if columns == 1 and len(lines) > 1:
                warn("{}: annotation wrapped at a head boundary to preserve 7-pt text.".format(selection.clip_id))
            elif columns == 2 and size < FONT_ANNOTATION:
                print("  Two-line annotation font: {:.2f} pt (width fit).".format(size), flush=True)
        except SkipClip as exc:
            warn("Skipping ({}) {}: {}".format(selection.block, selection.clip_id, exc))
    return clips


def calculate_layout(clips: Sequence[Clip], row_height: float, max_height: float,
                     demo: bool) -> Layout:
    if not clips:
        raise ValueError("No renderable clips remain; check the warnings, CSV paths, and frame roots.")
    width = WIDTH_IN * 72
    groups = sum(any(c.selection.block == b for c in clips) for b in BLOCKS)
    banner_h = 14.0 if demo else 0.0
    # There is only ONE timestamp row, above the first rendered strip.
    fixed = (MARGIN_TOP + MARGIN_BOTTOM + banner_h + groups * HEADER_H
             + max(0, groups - 1) * BLOCK_GAP + TIME_H)
    annotation_space = sum(ANNOTATION_GAP + len(c.lines) * ANNOTATION_LINE_H + ROW_GAP for c in clips)
    width_limit = min((width - 2 * MARGIN_X - (len(c.images) - 1) * FRAME_GAP)
                      / sum(c.aspects) for c in clips)
    row_limit = row_height * 72 - max(ANNOTATION_GAP + len(c.lines) * ANNOTATION_LINE_H + ROW_GAP for c in clips)
    preferred = min(width_limit, row_limit)
    allowed = (max_height * 72 - fixed - annotation_space) / len(clips)
    image_h = min(preferred, allowed)
    if image_h <= 0:
        raise ValueError("Selected rows cannot fit at the requested height without shrinking text. Reduce clips or increase --max-height/--row-height.")
    natural_h = fixed + annotation_space + len(clips) * preferred
    total_h = fixed + annotation_space + len(clips) * image_h
    if image_h < preferred - 0.01:
        print("Layout: natural height {:.3f} in -> {:.3f} in cap; common PHOTO height {:.3f} in; fonts unchanged (6/7/8 pt).".format(natural_h / 72, total_h / 72, image_h / 72), flush=True)
    if image_h < 18:
        warn("Photos are under 0.25 in high; fewer selected clips or a larger --max-height would improve legibility.")
    return Layout(width, total_h, image_h, natural_h, groups, banner_h)


def draw_figure(clips: Sequence[Clip], layout: Layout, outdir: Path, demo: bool,
                frames: int = 6) -> Tuple[Path, Path]:
    frame_names, time_labels = selected_frames(frames)
    rc = {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
          "ps.fonttype": 42, "text.usetex": False, "image.composite_image": False}
    stem = "fig3_qualitative_DEMO" if demo else "fig3_qualitative"
    png, pdf = outdir / (stem + ".png"), outdir / (stem + ".pdf")
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(WIDTH_IN, layout.height_pt / 72), dpi=DPI, facecolor="white")
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, layout.width_pt)
        ax.set_ylim(layout.height_pt, 0)  # Physical-point coordinates, origin at top left.
        ax.set_axis_off()
        y = MARGIN_TOP
        text_artists = []

        def label(x: float, baseline: float, text: str, size: float, color: str = BLACK,
                  weight: str = "normal", ha: str = "left") -> None:
            text_artists.append(ax.text(x, baseline, text, fontsize=size, color=color,
                                        weight=weight, ha=ha, va="baseline", clip_on=False, zorder=5))

        if demo:
            label(MARGIN_X, y + 8, "SYNTHETIC DEMO — layout only; not paper predictions", 8)
            y += layout.banner_h
        first = True
        for block, title in BLOCKS.items():
            members = [c for c in clips if c.selection.block == block]
            if not members:
                continue
            if not first:
                y += BLOCK_GAP
            label(MARGIN_X, y + FONT_TITLE, title, FONT_TITLE, weight="bold")
            y += HEADER_H
            for clip in members:
                widths = [layout.image_h * aspect for aspect in clip.aspects]
                strip_w = sum(widths) + (len(widths) - 1) * FRAME_GAP
                x = (layout.width_pt - strip_w) / 2
                if first:
                    tx = x
                    for frame_w, time in zip(widths, time_labels):
                        label(tx + frame_w / 2, y + FONT_TIME, time, FONT_TIME, ha="center")
                        tx += frame_w + FRAME_GAP
                    y += TIME_H
                    first = False
                for index, (image, frame_w) in enumerate(zip(clip.images, widths)):
                    target_h = max(1, math.ceil(layout.image_h * DPI / 72))
                    target_w = max(1, round(target_h * clip.aspects[index]))
                    resized = image.resize((target_w, target_h), LANCZOS)
                    ax.imshow(np.asarray(resized), extent=(x, x + frame_w, y + layout.image_h, y),
                              origin="upper", interpolation="none", aspect="auto", zorder=1)
                    if frame_names[index] == "t+0.0s.jpg":
                        ax.add_patch(Rectangle((x, y), frame_w, layout.image_h,
                                               fill=False, edgecolor=ORANGE, linewidth=1.5, zorder=3))
                    x += frame_w + FRAME_GAP
                y += layout.image_h + ANNOTATION_GAP
                for line in clip.lines:
                    tx = MARGIN_X
                    for run in line:
                        label(tx, y + FONT_ANNOTATION, run.text, FONT_ANNOTATION, run.color, run.weight)
                        tx += text_width(run.text, weight=run.weight)
                    y += ANNOTATION_LINE_H
                y += ROW_GAP
        assert abs(y + MARGIN_BOTTOM - layout.height_pt) < 1e-6
        canvas.draw()
        renderer = canvas.get_renderer()
        # Check real glyph extents rather than relying only on character counts.
        for artist in text_artists:
            box = artist.get_window_extent(renderer)
            if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5:
                raise ValueError("Text exceeds page boundary: " + artist.get_text())
        # Do not tight-crop: this keeps physical width EXACTLY 7.16 inches.
        fig.savefig(png, dpi=DPI, facecolor="white", bbox_inches=None, pad_inches=0)
        fig.savefig(pdf, dpi=DPI, facecolor="white", bbox_inches=None, pad_inches=0,
                    metadata={"Title": "Synthetic Fig. 3 layout test" if demo else "Qualitative examples",
                              "Subject": "Synthetic data, not paper results" if demo else "Selected three-head fault-estimation predictions"})
        fig.clear()
    return png, pdf


def calculate_grid_layout(clips: Sequence[Clip], row_height: float,
                          max_height: float, demo: bool) -> GridLayout:
    if not clips:
        raise ValueError("No renderable clips remain; check CSV paths and frame roots.")
    by_block = {b: [c for c in clips if c.selection.block == b] for b in BLOCKS}
    bands = []
    seen_column = [False, False]
    for pair in GRID_PAIRS:
        counts = tuple(len(by_block[b]) for b in pair)
        if not any(counts):
            continue  # Drop a completely empty band, but never exchange columns.
        titles = tuple(grid_title_lines(b) if by_block[b] else () for b in pair)
        timestamps = tuple(bool(counts[i]) and not seen_column[i] for i in range(2))
        for i in range(2):
            seen_column[i] = seen_column[i] or bool(counts[i])
        header_h = HEADER_H  # Fixed single-line height in every occupied panel.
        bands.append(GridBand(pair, titles, header_h, timestamps,
                              TIME_H if any(timestamps) else 0.0, max(counts)))
    n_rows = sum(band.row_count for band in bands)
    banner_h = 14.0 if demo else 0.0
    fixed = (MARGIN_TOP + GRID_BOTTOM_PAD + banner_h
             + sum(band.header_h + band.time_h for band in bands)
             + (len(bands) - 1) * GRID_BAND_GAP
             + sum(max(0, band.row_count - 1) * GRID_ROW_GAP for band in bands))
    caption_h = ANNOTATION_GAP + 2 * ANNOTATION_LINE_H
    available = GRID_COLUMN_W - 2 * GRID_PAD
    width_limit = min((available - (len(c.images) - 1) * FRAME_GAP) / sum(c.aspects)
                      for c in clips)
    preferred = min(width_limit, row_height * 72 - caption_h - GRID_ROW_GAP)
    allowed = (max_height * 72 - fixed) / n_rows - caption_h
    image_h = min(preferred, allowed)
    if image_h <= 0:
        raise ValueError("Grid cannot fit without shrinking text; reduce clips or increase "
                         "--row-height/--max-height.")
    natural_h = fixed + n_rows * (preferred + caption_h)
    total_h = fixed + n_rows * (image_h + caption_h)
    if image_h < preferred - 0.01:
        print("Grid height: {:.3f} -> {:.3f} in cap; only photos reduced; caption sizes unchanged."
              .format(natural_h / 72, total_h / 72), flush=True)
    if image_h < 18:
        warn("Grid photos are under 0.25 in high; reduce selected clips or increase --max-height.")
    print("Grid counts: a={}, b={}, c={}, d={}; occupied strip rows = {}; "
          "columns 3.50 in; gutter 0.16 in."
          .format(*(len(by_block[b]) for b in BLOCKS), n_rows), flush=True)
    return GridLayout(WIDTH_IN * 72, total_h, image_h, natural_h, banner_h, tuple(bands))


def draw_grid_figure(clips: Sequence[Clip], layout: GridLayout, outdir: Path,
                     demo: bool, frames: int = 6) -> Tuple[Path, Path]:
    frame_names, time_labels = selected_frames(frames)
    by_block = {b: [c for c in clips if c.selection.block == b] for b in BLOCKS}
    rc = {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
          "ps.fonttype": 42, "text.usetex": False, "image.composite_image": False}
    stem = "fig3_qualitative_DEMO" if demo else "fig3_qualitative"
    png, pdf = outdir / (stem + ".png"), outdir / (stem + ".pdf")
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(WIDTH_IN, layout.height_pt / 72), dpi=DPI, facecolor="white")
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, layout.width_pt)
        ax.set_ylim(layout.height_pt, 0)
        ax.set_axis_off()
        text_artists = []

        def label(x: float, baseline: float, text: str, size: float,
                  color: str = BLACK, weight: str = "normal", ha: str = "left",
                  column: Optional[int] = None) -> None:
            artist = ax.text(x, baseline, text, fontsize=size, color=color, weight=weight,
                             ha=ha, va="baseline", clip_on=False, zorder=5)
            text_artists.append((artist, column))

        y = MARGIN_TOP
        if demo:
            label(GRID_PAD, y + 8, "SYNTHETIC DEMO — layout only; not paper predictions", 8)
            y += layout.banner_h
        caption_h = ANNOTATION_GAP + 2 * ANNOTATION_LINE_H
        row_h = layout.image_h + caption_h + GRID_ROW_GAP
        for band_index, band in enumerate(layout.bands):
            if band_index:
                y += GRID_BAND_GAP
            for col, block in enumerate(band.blocks):
                x0 = col * (GRID_COLUMN_W + GRID_GUTTER)
                members = by_block[block]
                for j, title in enumerate(band.title_lines[col]):
                    label(x0 + GRID_PAD, y + j * HEADER_H + FONT_TITLE, title,
                          FONT_TITLE, weight="bold", column=col)
                first_photo_y = y + band.header_h + band.time_h
                for row_index, clip in enumerate(members):
                    image_y = first_photo_y + row_index * row_h
                    widths = [layout.image_h * aspect for aspect in clip.aspects]
                    strip_w = sum(widths) + (len(widths) - 1) * FRAME_GAP
                    x = x0 + GRID_PAD  # Every strip starts at the SAME left edge; never center.
                    if row_index == 0 and band.timestamps[col]:
                        tx = x
                        for frame_w, time in zip(widths, time_labels):
                            label(tx + frame_w / 2, first_photo_y - TIME_H + FONT_TIME,
                                  time, FONT_TIME, ha="center", column=col)
                            tx += frame_w + FRAME_GAP
                    for index, (image, frame_w) in enumerate(zip(clip.images, widths)):
                        target_h = max(1, math.ceil(layout.image_h * DPI / 72))
                        target_w = max(1, round(target_h * clip.aspects[index]))
                        resized = image.resize((target_w, target_h), LANCZOS)
                        ax.imshow(np.asarray(resized),
                                  extent=(x, x + frame_w, image_y + layout.image_h, image_y),
                                  origin="upper", interpolation="none", aspect="auto", zorder=1)
                        # The impact image is third in 4-frame mode, fourth in 6-frame mode.
                        if frame_names[index] == "t+0.0s.jpg":
                            ax.add_patch(Rectangle((x, image_y), frame_w, layout.image_h,
                                                   fill=False, edgecolor=ORANGE,
                                                   linewidth=1.5, zorder=3))
                        x += frame_w + FRAME_GAP
                    annotation_y = image_y + layout.image_h + ANNOTATION_GAP
                    if len(clip.lines) != 2:
                        raise ValueError("Grid captions must contain exactly two lines.")
                    for line in clip.lines:
                        tx = x0 + GRID_PAD
                        for run in line:
                            label(tx, annotation_y + FONT_ANNOTATION, run.text,
                                  clip.annotation_size, run.color, run.weight, column=col)
                            tx += text_width(run.text, clip.annotation_size, run.weight)
                        annotation_y += ANNOTATION_LINE_H
            y += (band.header_h + band.time_h
                  + band.row_count * (layout.image_h + caption_h)
                  + max(0, band.row_count - 1) * GRID_ROW_GAP)
        assert abs(y + GRID_BOTTOM_PAD - layout.height_pt) < 1e-6
        canvas.draw()
        renderer = canvas.get_renderer()
        # Check both page edges and the column edges, not just approximate widths.
        for artist, col in text_artists:
            box = artist.get_window_extent(renderer)
            if (box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5
                    or box.y1 > fig.bbox.height + 0.5):
                raise ValueError("Text exceeds page boundary: " + artist.get_text())
            if col is not None:
                left = col * (GRID_COLUMN_W + GRID_GUTTER)
                xleft = ax.transData.transform((left, 0))[0]
                xright = ax.transData.transform((left + GRID_COLUMN_W, 0))[0]
                if box.x0 < xleft - 0.5 or box.x1 > xright + 0.5:
                    raise ValueError("Text crosses column boundary: " + artist.get_text())
        # Preserve 7.16-in physical width; no tight-cropping or post-export scaling.
        fig.savefig(png, dpi=DPI, facecolor="white", bbox_inches=None, pad_inches=0)
        fig.savefig(pdf, dpi=DPI, facecolor="white", bbox_inches=None, pad_inches=0,
                    metadata={"Title": "Synthetic Fig. 3 grid test" if demo else "Qualitative examples",
                              "Subject": "Synthetic data, not paper results" if demo else
                              "Selected three-head fault-estimation predictions; two-column layout"})
        fig.clear()
    return png, pdf


def create_demo(outdir: Path) -> Tuple[Path, Path, Path, List[Path]]:
    """Exactly 3 synthetic clips x 6 gray JPEGs; no real data used."""
    root = outdir / "demo_inputs"
    frames = root / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    ids = ("demo_success", "demo_recovery", "demo_failure")
    settings = (
        (1, 1, 2, 2, 70, 70),   # (a): exact success.
        (1, 0, 0, 0, 25, 30),   # (b): situation wrong; rounded target 30 is correct.
        (7, 0, 1, 0, 50, 30),   # (c): equal-fault absorbed into B>A.
    )
    rows, qwen_rows = [], []
    font = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), 28)
    for i, (clip_id, setting) in enumerate(zip(ids, settings)):
        folder = frames / clip_id
        folder.mkdir(parents=True, exist_ok=True)
        for j, name in enumerate(FRAME_NAMES):
            gray = 150 + 10 * i + 5 * j
            image = Image.new("RGB", (640, 360), (gray, gray, gray))
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 619, 339), outline=(100, 100, 100), width=3)
            draw.text((38, 90), "SYNTHETIC CLIP {}".format(i + 1), fill=(35, 35, 35), font=font)
            draw.text((38, 145), "FRAME {} / 6".format(j + 1), fill=(35, 35, 35), font=font)
            draw.text((38, 200), "{}".format(name[:-4]), fill=(35, 35, 35), font=font)
            image.save(folder / name, quality=95)
        row = dict(zip(("sit_gt", "sit_pred", "cmp_gt", "cmp_pred", "ratio_gt", "ratio_pred"), setting))
        row.update(video=clip_id + ".mp4", ratio_abs_err=abs(setting[4] - setting[5]), within10=int(abs(setting[4] - setting[5]) <= 10))
        rows.append(row)
        qrow = dict(row)
        qrow["ratio_pred"] = 100 if i < 2 else 30
        qrow["ratio_abs_err"] = abs(qrow["ratio_gt"] - qrow["ratio_pred"])
        qrow["within10"] = int(qrow["ratio_abs_err"] <= 10)
        qwen_rows.append(qrow)
    columns = ["video", "sit_gt", "sit_pred", "cmp_gt", "cmp_pred", "ratio_gt", "ratio_pred", "ratio_abs_err", "within10"]
    iv_csv, q_csv = root / "demo_internvl.csv", root / "demo_qwen3.csv"
    pd.DataFrame(rows, columns=columns).to_csv(iv_csv, index=False)
    pd.DataFrame(qwen_rows, columns=columns).to_csv(q_csv, index=False)
    selection = root / "demo_clips.txt"
    selection.write_text("# Synthetic layout-only examples\na demo_success\nb demo_recovery\nc demo_failure\n", encoding="utf-8")
    # A separate optional selection reuses the same 18 images to exercise (d).
    (root / "demo_clips_with_d.txt").write_text(
        selection.read_text(encoding="utf-8") + "d demo_success\n", encoding="utf-8")
    return selection, iv_csv, q_csv, [frames]


def create_grid_demo(outdir: Path, frames_count: int) -> Tuple[Path, Path, Path, List[Path]]:
    """Eight synthetic clips: two per block; write only the selected frame files."""
    root = outdir / "grid_demo_inputs"
    frames_root = root / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    # (block, short ID, six prediction fields, Qwen3 ratio prediction)
    cases = (
        ("a", "A1", (7, 7, 2, 2, 100, 100), 80),
        ("a", "A2", (11, 11, 0, 0, 0, 0), 30),
        ("b", "B1", (4, 12, 2, 2, 60, 60), 20),
        ("b", "B2", (11, 7, 0, 0, 25, 30), 100),
        ("c", "C1", (0, 0, 1, 0, 50, 0), 100),
        ("c", "C2", (7, 0, 2, 0, 80, 30), 40),
        ("d", "D1", (5, 5, 1, 1, 50, 50), 90),
        ("d", "D2", (7, 7, 2, 2, 100, 100), 0),
    )
    frame_names, _ = selected_frames(frames_count)
    font = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), 36)
    rows, qwen_rows, config_lines = [], [], ["# SYNTHETIC: two clips per block; layout only"]
    for i, (block, short_id, setting, qratio) in enumerate(cases):
        clip_id = "demo_grid_" + short_id.lower()
        folder = frames_root / clip_id
        folder.mkdir(parents=True, exist_ok=True)
        config_lines.append(block + " " + clip_id)
        for j, name in enumerate(frame_names):
            gray = 160 + (i % 4) * 12 + j * 4
            image = Image.new("RGB", (640, 360), (gray, gray, gray))
            draw = ImageDraw.Draw(image)
            draw.rectangle((15, 15, 624, 344), outline=(110, 110, 110), width=3)
            draw.text((36, 66), "DUMMY " + short_id, fill=(35, 35, 35), font=font)
            draw.text((36, 136), name[:-4], fill=(35, 35, 35), font=font)
            draw.text((36, 216), "LAYOUT TEST", fill=(70, 70, 70), font=font)
            image.save(folder / name, quality=95)
        row = dict(zip(("sit_gt", "sit_pred", "cmp_gt", "cmp_pred", "ratio_gt", "ratio_pred"), setting))
        row.update(video=clip_id + ".mp4", ratio_abs_err=abs(setting[4] - setting[5]),
                   within10=int(abs(setting[4] - setting[5]) <= 10))
        rows.append(row)
        qrow = dict(row)
        qrow["ratio_pred"] = qratio
        qrow["ratio_abs_err"] = abs(qrow["ratio_gt"] - qratio)
        qrow["within10"] = int(qrow["ratio_abs_err"] <= 10)
        qwen_rows.append(qrow)
    iv_csv, q_csv = root / "demo_internvl.csv", root / "demo_qwen3.csv"
    pd.DataFrame(rows).to_csv(iv_csv, index=False)
    pd.DataFrame(qwen_rows).to_csv(q_csv, index=False)
    selection = root / "demo_clips.txt"
    selection.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    return selection, iv_csv, q_csv, [frames_root]


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="InternVL3 32f predictions CSV")
    parser.add_argument("--qwen-csv", default=DEFAULT_QWEN_CSV, help="Qwen3-VL 32f CSV; needed only for (d)")
    parser.add_argument("--clips", default=None, help="Selection file; default ./fig3_clips.txt, auto-initialized if absent")
    parser.add_argument("--frame-roots", nargs="+", default=list(DEFAULT_ROOTS), help="Search roots, in priority order")
    parser.add_argument("--columns", type=int, choices=(1, 2), default=1,
                        help="1: original vertical stack (default); 2: (a)|(b) above (c)|(d)")
    parser.add_argument("--frames", type=int, choices=(4, 6), default=6,
                        help="6: original frames (default); 4: -1.0, -0.5, +0.0, +0.5 seconds")
    parser.add_argument("--no-crop-letterbox", action="store_true",
                        help="Disable grid-only conservative exact-black border removal")
    parser.add_argument("--gamma", type=positive_float, default=1.0, help=">1 brightens; default 1.0")
    parser.add_argument("--outdir", default="./figs", help="Output directory")
    parser.add_argument("--row-height", type=positive_float, default=0.9, help="Preferred strip-row height in inches")
    parser.add_argument("--max-height", type=positive_float, default=9.2, help="Maximum figure height in inches")
    parser.add_argument("--demo", action="store_true",
                        help="Synthetic gray-frame test: 3 clips for one column, 8 for two columns")
    parser.add_argument("--write-default-clips", metavar="PATH", help="Write the exact default 14-clip config and exit; never overwrite")
    args = parser.parse_args(argv)
    try:
        if args.write_default_clips:
            target = expand_path(args.write_default_clips)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as out:
                out.write(DEFAULT_CLIPS)
            print("Wrote: {}".format(target))
            return 0
        outdir = expand_path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        if args.demo:
            if args.columns == 2:
                config, iv_csv, q_csv, roots = create_grid_demo(outdir, args.frames)
            else:
                config, iv_csv, q_csv, roots = create_demo(outdir)
            if args.clips:
                config = expand_path(args.clips)
            print("SYNTHETIC DEMO: layout validation only, not evidence for the paper.")
        else:
            config = expand_path(args.clips) if args.clips else Path("fig3_clips.txt")
            if args.clips is None and not config.exists():
                config.write_text(DEFAULT_CLIPS, encoding="utf-8")
                print("Initialized: {} (default 14 clips)".format(config.resolve()))
            iv_csv, q_csv = expand_path(args.csv), expand_path(args.qwen_csv)
            roots = [expand_path(p) for p in args.frame_roots]
        selected = read_selection(config)
        if not selected:
            raise ValueError("Selection file contains no valid rows: {}".format(config))
        print("Selection: {}\nInternVL3 CSV: {}\nGamma: {} (exponent = 1/gamma)".format(config.resolve(), iv_csv, args.gamma), flush=True)
        iv_index = read_predictions(iv_csv, "InternVL3")
        qwen_index = read_predictions(q_csv, "Qwen3-VL") if any(s.block == "d" for s in selected) else {}
        frame_names, _ = selected_frames(args.frames)
        print("Mode: {} column(s), {} frames: {}".format(args.columns, args.frames, ", ".join(frame_names)), flush=True)
        clips = collect_clips(selected, iv_index, qwen_index, roots, args.gamma,
                              args.row_height, args.columns, args.frames,
                              crop_letterbox=not args.no_crop_letterbox)
        if args.columns == 2:
            layout = calculate_grid_layout(clips, args.row_height, args.max_height, args.demo)
            png, pdf = draw_grid_figure(clips, layout, outdir, args.demo, args.frames)
        else:
            layout = calculate_layout(clips, args.row_height, args.max_height, args.demo)
            png, pdf = draw_figure(clips, layout, outdir, args.demo, args.frames)
        audit = []
        for c in clips:
            audit.append({"block": c.selection.block, "clip_id": c.selection.clip_id,
                          "frame_folder": str(c.folder), "annotation": joined_text(c.groups),
                          "ratio_gt_raw": c.iv.ratio_gt, "ratio_gt_display": quantize_gt(c.iv.ratio_gt),
                          "ratio_pred_internvl3": c.iv.ratio_pred,
                          "ratio_pred_qwen3": c.qwen.ratio_pred if c.qwen else "",
                          "block_condition_ok": c.condition_ok,
                          "columns": args.columns, "frames": args.frames,
                          "frame_files": " | ".join(frame_names),
                          "annotation_line1": "".join(r.text for r in c.lines[0]),
                          "annotation_line2": "".join(r.text for r in c.lines[1]) if len(c.lines) > 1 else "",
                          "annotation_font_pt": c.annotation_size,
                          "letterbox_top_px": c.letterbox_top_px,
                          "letterbox_bottom_px": c.letterbox_bottom_px,
                          "letterbox_status": c.letterbox_status})
        audit_path = png.with_name(png.stem + "_audit.tsv")
        pd.DataFrame(audit).to_csv(audit_path, sep="\t", index=False)
        print("Rendered {} / {} selected rows; skipped {}.".format(len(clips), len(selected), len(selected) - len(clips)))
        annotation_sizes = "/".join("{:g}".format(v) for v in sorted({c.annotation_size for c in clips}))
        print("Physical size: {:.2f} x {:.3f} in. Text: annotations {} pt; titles 8 pt; timestamps 6 pt."
              .format(WIDTH_IN, layout.height_pt / 72, annotation_sizes))
        print("Saved: {} (300 dpi)\nSaved: {}\nAudit: {}".format(png.resolve(), pdf.resolve(), audit_path.resolve()))
        return 0
    except (OSError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
