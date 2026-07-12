"""Compare detector outputs and build charts.

Reads every data/analysis_<detector>.csv from analyze.py, joins them on
post_url, and writes:

    data/comparison.csv - one row per post with each detector's scores
    data/charts/*.png   - agreement, consensus and supporting figures
    a short text report to stdout

Usage:
    python summarize.py
    python summarize.py --show   # also open interactive windows
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import config

# Friendly display names for the detectors.
_DISPLAY = {
    "statistical_stylometry": "Statistical stylometry",
    "fakespot_roberta": "Fakespot RoBERTa",
    "tmr_raid": "TMR RAID (RoBERTa)",          # legacy
    "hc3_chatgpt": "HC3 ChatGPT (RoBERTa)",    # legacy
}

# Muted colour palette (nicer than matplotlib's defaults for these charts).
_COLORS = {
    "both_ai": "#C45C26",
    "both_human": "#2F6F4E",
    "one_ai": "#C9A227",
    "both_mixed": "#6B7C8A",
    "disagree": "#8B5E3C",
    "def_human": "#2F6F4E",
    "might_ai": "#C9A227",
    "def_ai": "#C45C26",
    "a": "#3D5A80",
    "b": "#E07A5F",
}


def _label(name: str) -> str:
    return _DISPLAY.get(name, name)


def load_detector_results() -> dict[str, pd.DataFrame]:
    """Load every analysis_*.csv that matches a configured detector."""
    results: dict[str, pd.DataFrame] = {}
    for name in config.DETECTORS:
        path = config.analysis_csv_path(name)
        if not path.exists():
            print(f"  skip {name}: {path} not found", flush=True)
            continue
        df = pd.read_csv(path)
        results[name] = df
        print(f"  loaded {name}: {len(df)} posts from {path.name}", flush=True)
    return results


def build_comparison(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join detector results on post_url with detector-prefixed columns."""
    if len(results) < 2:
        raise ValueError(
            f"Need at least 2 detector result files; found {len(results)}. "
            "Run analyze.py first."
        )

    names = list(results.keys())
    base_cols = ["post_url", "author_name", "keyword", "word_count"]
    score_cols = ["p_ai_full", "pct_ai", "n_sentences", "n_sent_ai", "verdict"]

    merged: pd.DataFrame | None = None
    for name, df in results.items():
        piece = df[base_cols + score_cols].copy()
        piece = piece.rename(columns={c: f"{c}__{name}" for c in score_cols})
        if merged is None:
            merged = piece
        else:
            # Keep first copy of shared metadata columns.
            merged = merged.merge(
                piece.drop(columns=["author_name", "keyword", "word_count"],
                           errors="ignore"),
                on="post_url",
                how="inner",
            )

    assert merged is not None
    a, b = names[0], names[1]
    va, vb = f"verdict__{a}", f"verdict__{b}"

    # Agreement bucket. "Only one AI" = exactly one detector says AI.
    def agreement_row(row: pd.Series) -> str:
        if row[va] == "AI" and row[vb] == "AI":
            return "Both AI"
        if row[va] == "Human" and row[vb] == "Human":
            return "Both Human"
        if (row[va] == "AI") ^ (row[vb] == "AI"):
            return "Only one AI"
        if row[va] == "Mixed" and row[vb] == "Mixed":
            return "Both Mixed"
        return "Disagree (other)"

    merged["agreement"] = merged.apply(agreement_row, axis=1)

    # Consensus bucket: both Human -> "Definitely human", both AI ->
    # "Definitely AI", anything else (Mixed, split votes) -> "Might be AI".
    def consensus_row(row: pd.Series) -> str:
        if row[va] == "Human" and row[vb] == "Human":
            return "Definitely human"
        if row[va] == "AI" and row[vb] == "AI":
            return "Definitely AI"
        return "Might be AI"

    merged["consensus"] = merged.apply(consensus_row, axis=1)

    merged["mean_p_ai"] = merged[[f"p_ai_full__{a}", f"p_ai_full__{b}"]].mean(axis=1)
    merged["mean_pct_ai"] = merged[[f"pct_ai__{a}", f"pct_ai__{b}"]].mean(axis=1)
    merged["detector_a"] = a
    merged["detector_b"] = b
    return merged


def _save(fig: plt.Figure, name: str, show: bool) -> Path:
    out = config.CHARTS_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return out


def chart_agreement(df: pd.DataFrame, show: bool) -> Path:
    """Bar chart: both AI / both Human / only one AI / etc."""
    order = ["Both AI", "Both Human", "Only one AI", "Both Mixed", "Disagree (other)"]
    counts = df["agreement"].value_counts()
    labels = [o for o in order if o in counts.index]
    values = [int(counts[o]) for o in labels]
    colors = {
        "Both AI": _COLORS["both_ai"],
        "Both Human": _COLORS["both_human"],
        "Only one AI": _COLORS["one_ai"],
        "Both Mixed": _COLORS["both_mixed"],
        "Disagree (other)": _COLORS["disagree"],
    }

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(labels, values, color=[colors[l] for l in labels], width=0.65)
    ax.set_title("Model agreement on AI vs human", fontsize=14, pad=12)
    ax.set_ylabel("Number of posts")
    ax.set_ylim(0, max(values) * 1.18 if values else 1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.02,
                f"{v}\n({100 * v / len(df):.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "01_agreement.png", show)


def chart_consensus_pie(df: pd.DataFrame, show: bool) -> Path:
    """Pie: definitely human / might be AI / definitely AI."""
    order = ["Definitely human", "Might be AI", "Definitely AI"]
    counts = df["consensus"].value_counts()
    labels = [o for o in order if o in counts.index]
    values = [int(counts[o]) for o in labels]
    colors = {
        "Definitely human": _COLORS["def_human"],
        "Might be AI": _COLORS["might_ai"],
        "Definitely AI": _COLORS["def_ai"],
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=[colors[l] for l in labels],
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * len(df) / 100))})",
        startangle=90,
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
        textprops={"fontsize": 11},
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Consensus: how sure are both models?", fontsize=14, pad=14)
    return _save(fig, "02_consensus_pie.png", show)


def chart_verdicts_side_by_side(df: pd.DataFrame, show: bool) -> Path:
    """Grouped bars of each detector's AI / Human / Mixed counts."""
    a = df["detector_a"].iloc[0]
    b = df["detector_b"].iloc[0]
    verdicts = ["Human", "Mixed", "AI"]
    ca = df[f"verdict__{a}"].value_counts()
    cb = df[f"verdict__{b}"].value_counts()
    ya = [int(ca.get(v, 0)) for v in verdicts]
    yb = [int(cb.get(v, 0)) for v in verdicts]

    x = range(len(verdicts))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], ya, width, label=_label(a),
           color=_COLORS["a"])
    ax.bar([i + width / 2 for i in x], yb, width, label=_label(b),
           color=_COLORS["b"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(verdicts)
    ax.set_ylabel("Number of posts")
    ax.set_title("Verdict breakdown by detector", fontsize=14, pad=12)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (va, vb) in enumerate(zip(ya, yb)):
        ax.text(i - width / 2, va, str(va), ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, vb, str(vb), ha="center", va="bottom", fontsize=8)
    return _save(fig, "03_verdicts_by_detector.png", show)


def chart_score_scatter(df: pd.DataFrame, show: bool) -> Path:
    """Scatter of whole-post P(AI) from model A vs model B."""
    a = df["detector_a"].iloc[0]
    b = df["detector_b"].iloc[0]
    xa = df[f"p_ai_full__{a}"]
    yb = df[f"p_ai_full__{b}"]

    color_map = {
        "Definitely human": _COLORS["def_human"],
        "Might be AI": _COLORS["might_ai"],
        "Definitely AI": _COLORS["def_ai"],
    }
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for label, color in color_map.items():
        mask = df["consensus"] == label
        if not mask.any():
            continue
        ax.scatter(xa[mask], yb[mask], c=color, s=18, alpha=0.55,
                   label=label, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1,
            label="Perfect agreement")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"P(AI) — {_label(a)}")
    ax.set_ylabel(f"P(AI) — {_label(b)}")
    ax.set_title("Whole-post AI probability: model vs model", fontsize=14, pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "04_score_scatter.png", show)


def chart_pct_ai_hist(df: pd.DataFrame, show: bool) -> Path:
    """Overlapping histograms of sentence-level % AI for each detector."""
    a = df["detector_a"].iloc[0]
    b = df["detector_b"].iloc[0]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bins = list(range(0, 105, 5))
    ax.hist(df[f"pct_ai__{a}"], bins=bins, alpha=0.55, color=_COLORS["a"],
            label=_label(a), edgecolor="white")
    ax.hist(df[f"pct_ai__{b}"], bins=bins, alpha=0.55, color=_COLORS["b"],
            label=_label(b), edgecolor="white")
    ax.axvline(30, color="#2F6F4E", linestyle=":", linewidth=1.2, label="Human ≤ 30%")
    ax.axvline(70, color="#C45C26", linestyle=":", linewidth=1.2, label="AI ≥ 70%")
    ax.set_xlabel("% of post scored as AI (sentence-weighted)")
    ax.set_ylabel("Number of posts")
    ax.set_title("Distribution of sentence-level AI share", fontsize=14, pad=12)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "05_pct_ai_distribution.png", show)


def chart_by_keyword(df: pd.DataFrame, show: bool) -> Path:
    """Stacked bars: consensus mix within each scrape keyword."""
    order = ["Definitely human", "Might be AI", "Definitely AI"]
    colors = [_COLORS["def_human"], _COLORS["might_ai"], _COLORS["def_ai"]]
    ct = (
        df.groupby(["keyword", "consensus"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
    )
    # Sort keywords by volume.
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = None
    for col, color in zip(order, colors):
        vals = ct[col].values
        ax.bar(ct.index.astype(str), vals, bottom=bottom, color=color,
               label=col, width=0.6)
        bottom = vals if bottom is None else bottom + vals
    ax.set_ylabel("Number of posts")
    ax.set_title("Consensus by search keyword", fontsize=14, pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, "06_consensus_by_keyword.png", show)


def chart_agreement_matrix(df: pd.DataFrame, show: bool) -> Path:
    """Heatmap-style matrix of verdict A × verdict B counts."""
    a = df["detector_a"].iloc[0]
    b = df["detector_b"].iloc[0]
    order = ["Human", "Mixed", "AI"]
    mat = (
        pd.crosstab(df[f"verdict__{a}"], df[f"verdict__{b}"])
        .reindex(index=order, columns=order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat.values, cmap="YlOrBr", aspect="equal")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticklabels(order)
    ax.set_xlabel(_label(b))
    ax.set_ylabel(_label(a))
    ax.set_title("Verdict cross-tab (counts)", fontsize=14, pad=12)
    for i in range(len(order)):
        for j in range(len(order)):
            val = int(mat.values[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > mat.values.max() * 0.55 else "#222",
                    fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, "07_verdict_matrix.png", show)


def print_report(df: pd.DataFrame) -> None:
    a = df["detector_a"].iloc[0]
    b = df["detector_b"].iloc[0]
    n = len(df)

    print("\n" + "=" * 60)
    print(f"CROSS-DETECTOR SUMMARY  ({n} posts)")
    print(f"  A = {_label(a)}")
    print(f"  B = {_label(b)}")
    print("=" * 60)

    print("\nAgreement")
    for label, count in df["agreement"].value_counts().items():
        print(f"  {label:22s}  {count:5d}  ({100 * count / n:5.1f}%)")

    print("\nConsensus")
    for label in ["Definitely human", "Might be AI", "Definitely AI"]:
        count = int((df["consensus"] == label).sum())
        print(f"  {label:22s}  {count:5d}  ({100 * count / n:5.1f}%)")

    print("\nPer-detector verdicts")
    for name in (a, b):
        print(f"  {_label(name)}:")
        for v, c in df[f"verdict__{name}"].value_counts().items():
            print(f"    {v:8s}  {c:5d}  ({100 * c / n:5.1f}%)")

    # Top "most AI-like" posts by mean score (handy for spot-checks).
    top = df.nlargest(5, "mean_p_ai")[
        ["author_name", "keyword", "mean_p_ai", "consensus", "post_url"]
    ]
    print("\nHighest mean P(AI) posts (spot-check candidates)")
    for _, row in top.iterrows():
        print(f"  {row['mean_p_ai']:.3f}  [{row['consensus']}]  "
              f"{row['author_name']}  ({row['keyword']})")
        print(f"         {row['post_url']}")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarise and chart cross-detector AI/human results.")
    p.add_argument("--show", action="store_true",
                   help="Open interactive matplotlib windows as well as saving PNGs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("Loading detector results ...", flush=True)
    results = load_detector_results()
    if len(results) < 2:
        print(
            "ERROR: need at least two analysis_*.csv files. "
            "Run `python analyze.py` first.",
            file=sys.stderr,
        )
        return 2

    print("Building comparison table ...", flush=True)
    comparison = build_comparison(results)
    comparison.to_csv(config.COMPARISON_CSV, index=False, encoding="utf-8")
    print(f"  wrote {config.COMPARISON_CSV}", flush=True)

    print("Building charts ...", flush=True)
    chart_agreement(comparison, args.show)
    chart_consensus_pie(comparison, args.show)
    chart_verdicts_side_by_side(comparison, args.show)
    chart_score_scatter(comparison, args.show)
    chart_pct_ai_hist(comparison, args.show)
    chart_by_keyword(comparison, args.show)
    chart_agreement_matrix(comparison, args.show)

    print_report(comparison)
    print(f"Charts directory -> {config.CHARTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
