"""Small scientific figures for the decoded BIB/LLP first-look tables.

The caller owns decoding, associations and unit validation.  This module uses
stored values, retains all finite negative timing values, and does not infer
truth labels or cell-to-jet links.  Each PNG has a matching standalone PDF.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SAMPLES = ("LLP sample", "Unpaired sample")
COLORS = {"LLP sample": "#2563a6", "Unpaired sample": "#c66b12"}


def _values(frame, column):
    """Preserve row order while exposing numeric missing values as NaN."""
    return frame[column].to_numpy(dtype=float, na_value=np.nan)


def _linear_bounds(arrays):
    nonempty = [x for x in arrays if len(x)]
    if not nonempty:
        return (-1.0, 1.0)
    values = np.concatenate(nonempty)
    lo, hi = float(values.min()), float(values.max())
    padding = 0.04 * (hi - lo) if hi > lo else max(abs(lo) * 0.05, 0.5)
    return (lo - padding, hi + padding)


def _bins(arrays, *, positive_log=False, count=35):
    nonempty = [x for x in arrays if len(x)]
    if not nonempty:
        return np.geomspace(1.0, 10.0, count + 1) if positive_log else np.linspace(-1, 1, count + 1)
    values = np.concatenate(nonempty)
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        if positive_log:
            lo, hi = lo / 1.2, hi * 1.2
        else:
            lo, hi = _linear_bounds([values])
    return np.geomspace(lo, hi, count + 1) if positive_log else np.linspace(lo, hi, count + 1)


def _save(fig, output_dir, stem):
    png = output_dir / (stem + ".png")
    pdf = output_dir / (stem + ".pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(png)


def _histogram(ax, values, bins, sample):
    if len(values):
        ax.hist(values, bins=bins, weights=np.full(len(values), 1.0 / len(values)),
                histtype="step", linewidth=1.8, color=COLORS[sample],
                label=f"{sample} (N={len(values):,})")
    else:
        ax.plot([], [], color=COLORS[sample], label=f"{sample} (N=0)")


def _identifier(row, column):
    value = row.get(column)
    if value is None:
        return "unknown"
    # Never route event identifiers through floating-point conversion.
    return str(value)


def make_plots(jets, cells, output_dir):
    """Return ``{figure_stem: absolute_png_path}``; also save matching PDFs.

    ``jets`` contains decoded jet rows with sample, jet_pt, jet_eta,
    jet_time_ATLAS and jet_time_spread.  ``cells`` contains all decoded cells
    from at most one stored row per sample, plus source_file, row_index,
    runNumber, lumiBlock, eventNumber and jet_count metadata.  Missing numeric
    values are counted as nonfinite exclusions.  No input frames are modified.
    """
    jet_required = {"sample", "jet_pt", "jet_eta", "jet_time_ATLAS", "jet_time_spread"}
    cell_required = {"sample", "cell_z", "cell_time", "source_file", "row_index",
                     "runNumber", "lumiBlock", "eventNumber", "jet_count"}
    for name, frame, required in (("jets", jets, jet_required), ("cells", cells, cell_required)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} table lacks columns: {sorted(missing)}")
        unexpected = set(frame["sample"].dropna().unique()) - set(SAMPLES)
        if unexpected or frame["sample"].isna().any():
            raise ValueError(f"{name} table has unknown or missing sample labels: {unexpected}")
    for sample in SAMPLES:
        group = cells.loc[cells["sample"] == sample]
        if len(group) and len(group[["source_file", "row_index"]].drop_duplicates()) != 1:
            raise ValueError(f"Cell figure expects one stored row for {sample}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}
    style = {"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
             "figure.titlesize": 15, "legend.fontsize": 9, "axes.spines.top": False,
             "axes.spines.right": False, "axes.axisbelow": True}
    with plt.rc_context(style):
        # 1. Comparable bins; each source is normalized to unit total bin weight.
        kin = {}
        for sample in SAMPLES:
            group = jets.loc[jets["sample"] == sample]
            pt, eta = _values(group, "jet_pt"), _values(group, "jet_eta")
            pt_finite = np.isfinite(pt)
            kin[sample] = {
                "total": len(group), "pt": pt[pt_finite & (pt > 0)],
                "eta": eta[np.isfinite(eta)],
                "pt_nonfinite": int((~pt_finite).sum()),
                "pt_nonpositive": int((pt_finite & (pt <= 0)).sum()),
                "eta_nonfinite": int((~np.isfinite(eta)).sum()),
            }
        pt_bins = _bins([kin[s]["pt"] for s in SAMPLES], positive_log=True)
        eta_bins = _bins([kin[s]["eta"] for s in SAMPLES])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
        fig.subplots_adjust(left=0.075, right=0.98, top=0.82, bottom=0.28, wspace=0.24)
        fig.suptitle("Stored jet kinematics by source")
        for sample in SAMPLES:
            _histogram(axes[0], kin[sample]["pt"], pt_bins, sample)
            _histogram(axes[1], kin[sample]["eta"], eta_bins, sample)
        axes[0].set(xscale="log", xlabel="jet_pt (stored units)", ylabel="Fraction of included jets per bin",
                    title="Positive jet_pt; logarithmic x axis")
        axes[1].set(xlabel="jet_eta (stored value)", ylabel="Fraction of included jets per bin",
                    title="Finite jet_eta; linear x axis")
        for ax in axes:
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
            ax.legend(loc="best", frameon=False)
        notes = ["Common bins per panel; each source sums to 1. Bin fractions are not physical rates."]
        for sample in SAMPLES:
            k = kin[sample]
            notes.append(f"{sample}: {k['total']:,} decoded jets; jet_pt excluded {k['pt_nonfinite']:,} nonfinite"
                         f" + {k['pt_nonpositive']:,} finite <= 0; jet_eta excluded {k['eta_nonfinite']:,} nonfinite.")
        fig.text(0.075, 0.065, "\n".join(notes), ha="left", va="bottom", fontsize=9, linespacing=1.5)
        plots["01_jet_kinematics"] = _save(fig, output_dir, "01_jet_kinematics")

        # 2. All finite pairs remain visible on shared full-range axes.
        timing = {}
        for sample in SAMPLES:
            group = jets.loc[jets["sample"] == sample]
            time = _values(group, "jet_time_ATLAS")
            spread = _values(group, "jet_time_spread")
            finite = np.isfinite(time) & np.isfinite(spread)
            timing[sample] = (time[finite], spread[finite], int((~finite).sum()), len(group))
        time_bounds = _linear_bounds([timing[s][0] for s in SAMPLES])
        spread_bounds = _linear_bounds([timing[s][1] for s in SAMPLES])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.81, bottom=0.23, wspace=0.14)
        fig.suptitle("Stored jet timing and spread")
        for ax, sample in zip(axes, SAMPLES):
            time, spread, excluded, total = timing[sample]
            ax.scatter(time, spread, s=8, alpha=0.3, color=COLORS[sample],
                       edgecolors="none", rasterized=True)
            ax.set(xlabel="jet_time_ATLAS (stored units)",
                   xlim=time_bounds, ylim=spread_bounds)
            ax.set_title(sample, pad=30)
            ax.grid(alpha=0.2)
            ax.text(0.02, 1.015, f"N={len(time):,} finite pairs / {total:,}; excluded {excluded:,}",
                    transform=ax.transAxes, fontsize=9, ha="left", va="bottom")
        axes[0].set_ylabel("jet_time_spread (stored units)")
        fig.text(0.08, 0.055,
                 "Shared axes cover every finite pair. Negative stored times/spreads are retained.\n"
                 "Units, calibrations and source selections remain unconfirmed; these are source comparisons.",
                 fontsize=9, ha="left", va="bottom", linespacing=1.5)
        plots["02_jet_timing"] = _save(fig, output_dir, "02_jet_timing")

        # 3. These are row-level cell displays, without any inferred jet link.
        examples = {}
        for sample in SAMPLES:
            group = cells.loc[cells["sample"] == sample]
            z, time = _values(group, "cell_z"), _values(group, "cell_time")
            finite = np.isfinite(z) & np.isfinite(time)
            examples[sample] = (group, z[finite], time[finite], int((~finite).sum()))
        z_bounds = _linear_bounds([examples[s][1] for s in SAMPLES])
        cell_time_bounds = _linear_bounds([examples[s][2] for s in SAMPLES])
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.73, bottom=0.23, wspace=0.14)
        fig.suptitle("All cells in one stored row per source", y=0.98)
        for ax, sample in zip(axes, SAMPLES):
            group, z, time, excluded = examples[sample]
            if len(group):
                row = group.iloc[0]
                identity = (f"run {_identifier(row, 'runNumber')}, LB {_identifier(row, 'lumiBlock')}, "
                            f"event {_identifier(row, 'eventNumber')}")
                detail = (f"stored row {_identifier(row, 'row_index')}; "
                          f"jet_count={_identifier(row, 'jet_count')}")
                title = f"{sample}\n{identity}\n{detail}"
            else:
                title = f"{sample}\nNo decoded cell row supplied"
            ax.set_title(title, fontsize=10, pad=12)
            ax.scatter(z, time, s=9, alpha=0.45, color=COLORS[sample],
                       edgecolors="none", rasterized=True)
            ax.set(xlabel="cell_z (stored units)", xlim=z_bounds, ylim=cell_time_bounds)
            ax.grid(alpha=0.2)
            ax.text(0, -0.19, f"Stored cells: {len(group):,}; finite pairs: {len(z):,}; excluded: {excluded:,}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9)
        axes[0].set_ylabel("cell_time (stored units)")
        fig.text(0.08, 0.035,
                 "All cells in one stored row; jet links and units unconfirmed. No cells are assigned to a jet here.\n"
                 "Only nonfinite coordinate/time pairs are excluded. Full finite ranges and negative times are retained.",
                 fontsize=9, ha="left", va="bottom", linespacing=1.5)
        plots["03_cell_timing"] = _save(fig, output_dir, "03_cell_timing")
    return plots
