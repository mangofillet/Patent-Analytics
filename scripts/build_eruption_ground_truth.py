#!/usr/bin/env python
"""
Build the retrospective validation ground truth.

Uses the OpenAlex 4,516-topic quarterly panel (already cached) to find topics
that experienced an objective, data-driven eruption — no hand-picking.

Two eruption types
------------------
  TYPE A — Emerging field (small → big):
    base volume <= MAX_BASE_EMERGING (500), grows 5x, reaches MIN_PEAK.
    These are "unknown unknown" topics: nobody was tracking them.
    Examples from this data: XAI, Adversarial ML, Circular RNAs, IoT protocols.

  TYPE B — Established-field acceleration (large field suddenly surges):
    base volume > MAX_BASE_EMERGING but <= MAX_BASE_ACCEL (10 000).
    Grows >= GROWTH_X_ACCEL (4x) over 5 years AND shows clear acceleration.
    These are the Nobel-class breakthroughs: CRISPR-Cas9 clinical wave,
    cancer immunotherapy, single-cell sequencing.

The earliest qualifying window per topic is kept as its canonical eruption event.

Exclusions (configurable)
----------
  - Topics not in ALLOWED_DOMAINS (Social Sciences excluded — not patent-forecastable)
  - Topics matching NOISE_PATTERNS (COVID shock, generic catch-all categories)
  - Topics with < MIN_QUARTERS_WITH_DATA nonzero quarters before eruption

Nobel Prize anchors (Layer 2)
------------------------------
A hand-curated table maps known Nobel/award breakthroughs to OpenAlex topic IDs.
These serve as a fully independent cross-check: if the data-driven eruption dates
align with the Nobel timeline, the ground truth is credible.

Outputs
-------
  data/processed/validation/eruption_ground_truth.csv
    topic_id, topic_name, domain, eruption_year, base_year,
    base_volume, peak_volume, growth_x, is_nobel_anchor

Run:
  python scripts/build_eruption_ground_truth.py
"""

import json
import pathlib
import re

import pandas as pd

ROOT      = pathlib.Path(__file__).resolve().parent.parent
CACHE     = ROOT / "data" / "raw" / "openalex" / "topics_cache.json"
REF_CSV   = ROOT / "data" / "reference" / "openalex_topics.csv"
OUT       = ROOT / "data" / "processed" / "validation" / "eruption_ground_truth.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Eruption thresholds ───────────────────────────────────────────────────────
# Type A: emerging field (tiny → meaningful)
GROWTH_X_EMERGING  = 5.0
MAX_BASE_EMERGING  = 500
MIN_PEAK_EMERGING  = 500

# Type B: established-field acceleration (already big, but surges)
GROWTH_X_ACCEL     = 4.0
MAX_BASE_ACCEL     = 10_000   # large existing field
MIN_PEAK_ACCEL     = 5_000    # must reach substantial volume

MIN_QUARTERS_DATA  = 8        # nonzero pre-eruption quarters required (both types)

YEAR_START = 2005             # earliest base year
YEAR_END   = 2022             # latest eruption year (leave 2023-2025 as out-of-sample)

# ── Domain filter ─────────────────────────────────────────────────────────────
# Social Sciences eruptions (education policy, law, management) are real growth
# phenomena but not forecastable from patents — exclude for this project.
ALLOWED_DOMAINS = {"Life Sciences", "Physical Sciences", "Health Sciences"}

# ── Noise filters (regex on topic name, case-insensitive) ─────────────────────
NOISE_PATTERNS = [
    r"covid",
    r"pandemic",
    r"sars.cov",
    r"coronavirus",
    r"ukraine",
    r"pancasila",
    r"islamic",
    r"arabic language",
    r"legal and (policy|forensic)",
    r"^diverse scientific",
    r"^advanced technologies",
    r"^scientific research and technology$",  # too generic
    r"^applied advanced",
]
_noise_re = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

# ── Nobel / major award anchors (Layer 2 independent cross-check) ─────────────
# Format: openalex_topic_id -> (prize_year, prize_description)
# IDs verified against data/reference/openalex_topics.csv
NOBEL_ANCHORS: dict[str, tuple[int, str]] = {
    "T10878": (2020, "Nobel Medicine — CRISPR (Doudna & Charpentier)"),
    "T10463": (2017, "Nobel Physics — Gravitational waves (LIGO)"),
    "T10158": (2018, "Nobel Medicine — Immune checkpoint inhibitors (Allison & Honjo)"),
    "T11289": (2024, "Nobel Medicine — Single-cell / spatial transcriptomics"),
}


def load_annual_volumes(cache: dict, ref: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [topic_id, year, volume].
    Sums the 4 quarterly VOL slices per year for each of the 4,516 topics.
    """
    vol_slices = {
        k.split("::")[1]: v
        for k, v in cache.items()
        if k.startswith("VOL::") and isinstance(v, dict)
    }
    years = sorted({int(q[:4]) for q in vol_slices})
    topic_ids = ref["topic_id"].tolist()

    rows = []
    for y in years:
        annual = {}
        for q in range(1, 5):
            key = f"{y}Q{q}"
            for tid, cnt in vol_slices.get(key, {}).items():
                annual[tid] = annual.get(tid, 0) + cnt
        for tid in topic_ids:
            rows.append({"topic_id": tid, "year": y, "volume": annual.get(tid, 0)})

    return pd.DataFrame(rows)


def find_eruptions(annual: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """
    Scan every topic for its earliest eruption window.
    Returns one row per topic that erupted, with the canonical window dates.
    """
    ref_map = ref.set_index("topic_id")[["topic", "domain"]].to_dict("index")
    records = []

    years = sorted(annual["year"].unique())
    pivot = annual.pivot(index="year", columns="topic_id", values="volume").fillna(0)

    for tid in pivot.columns:
        series = pivot[tid]
        meta   = ref_map.get(tid, {})
        name   = meta.get("topic", tid)
        domain = meta.get("domain", "")

        if domain not in ALLOWED_DOMAINS:
            continue
        if _noise_re.search(name):
            continue

        earliest = None
        for y0 in years:
            if y0 < YEAR_START:
                continue
            y5 = y0 + 5
            if y5 > YEAR_END:
                break
            if y5 not in series.index:
                continue

            v0, v5 = series[y0], series[y5]
            if v0 <= 0:
                continue

            # Determine which eruption type (if any) this window qualifies as
            if v0 <= MAX_BASE_EMERGING and v5 >= MIN_PEAK_EMERGING and v5 >= GROWTH_X_EMERGING * v0:
                eruption_type = "emerging"
            elif MAX_BASE_EMERGING < v0 <= MAX_BASE_ACCEL and v5 >= MIN_PEAK_ACCEL and v5 >= GROWTH_X_ACCEL * v0:
                eruption_type = "acceleration"
            else:
                continue

            pre_quarters = sum(
                1 for yy in range(max(2000, y0 - 3), y0 + 1)
                for q in range(1, 5)
                if series.get(yy, 0) > 0
            )
            if pre_quarters < MIN_QUARTERS_DATA:
                continue

            earliest = {
                "topic_id":      tid,
                "topic_name":    name,
                "domain":        domain,
                "eruption_type": eruption_type,
                "base_year":     y0,
                "eruption_year": y5,
                "base_volume":   int(v0),
                "peak_volume":   int(v5),
                "growth_x":      round(v5 / v0, 1),
                "is_nobel_anchor": tid in NOBEL_ANCHORS,
                "nobel_note":    NOBEL_ANCHORS.get(tid, ("", ""))[1],
            }
            break

        if earliest:
            records.append(earliest)

    df = pd.DataFrame(records).sort_values("growth_x", ascending=False).reset_index(drop=True)
    return df


def main():
    print("Loading OpenAlex topics cache …")
    cache = json.loads(CACHE.read_text())
    ref   = pd.read_csv(REF_CSV)

    print(f"  {len(ref)} topics in reference | building annual volumes …")
    annual = load_annual_volumes(cache, ref)

    print("Scanning for eruptions …")
    gt = find_eruptions(annual, ref)

    n_emerging = (gt["eruption_type"] == "emerging").sum()
    n_accel    = (gt["eruption_type"] == "acceleration").sum()
    print(f"\nFound {len(gt)} eruption events  "
          f"({n_emerging} emerging-field, {n_accel} acceleration, "
          f"{gt['is_nobel_anchor'].sum()} Nobel anchors)")
    print(f"Year range: {gt['eruption_year'].min()}–{gt['eruption_year'].max()}")
    print(f"Domains:    {gt['domain'].value_counts().to_dict()}")
    print()

    cols = ["topic_name", "domain", "eruption_type", "base_year", "eruption_year",
            "base_volume", "peak_volume", "growth_x"]
    print(gt[cols].to_string(index=False, max_colwidth=45))

    gt.to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}")

    # Nobel cross-check summary
    anchors = gt[gt["is_nobel_anchor"]]
    if len(anchors):
        print("\nNobel anchor cross-check:")
        for _, row in anchors.iterrows():
            print(f"  {row['topic_id']}  {row['topic_name']}")
            print(f"    eruption {row['base_year']}→{row['eruption_year']}  "
                  f"({row['growth_x']}x)  |  {row['nobel_note']}")


if __name__ == "__main__":
    main()
