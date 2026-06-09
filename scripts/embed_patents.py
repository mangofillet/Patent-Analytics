#!/usr/bin/env python
"""
Stage 1: Embed USPTO patent titles with sentence-transformers.

Reads g_patent.tsv.zip, filters to granted utility patents (wipo_kind=B2),
and batch-encodes every title into a 384-d semantic embedding using
all-MiniLM-L6-v2. Saves one compressed .npz per year so Stage 2 (clustering)
can work a year-window at a time without loading everything into memory.

Output layout:
  data/processed/embeddings/
    YYYY.npz   — keys: 'ids' (str), 'dates' (str), 'titles' (str), 'embeddings' (float32 384-d)

Run (CPU, recommended):
  python scripts/embed_patents.py --years 1996 2024 --batch-size 64
Run (GPU):
  python scripts/embed_patents.py --years 1976 2025 --batch-size 512

Data range: 1976–2025 (9.4M patents total). Recommended floor:
  1996+ — internet/genomics era, modern language, best cluster quality (~4.7M patents)
  1990+ — adds early biotech history, slightly older language (~5.3M patents)
  1976+ — full history; pre-1990 patents have limited relevance to modern topics

Time estimate on CPU (batch=64):
  1996–2024 (29 years) ~ 4–6 hours
  2000–2024 (25 years) ~ 3–5 hours
Time estimate on GPU (batch=512): ~1–2 min/year
"""

import argparse
import csv
import io
import os
import pathlib
import time
import zipfile
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT         = pathlib.Path(__file__).resolve().parent.parent
PATENTS      = ROOT / "data" / "raw" / "patents" / "g_patent.tsv.zip"
OUT_DIR      = ROOT / "data" / "processed" / "embeddings"
MODEL_ID     = "all-MiniLM-L6-v2"   # 384-d, fast, good for short text
WIPO_GRANTED = {"B2", "B1"}          # granted utility patents only


def parse_args():
    p = argparse.ArgumentParser(
        description="Embed USPTO patent titles into 384-d vectors (one .npz per year)."
    )
    p.add_argument("--years", nargs=2, type=int, default=[1996, 2024],
                   metavar=("START", "END"),
                   help="Year range to embed (inclusive). Default: 1996–2024. "
                        "Data goes back to 1976 but pre-1996 language quality drops off. "
                        "Use 1990 2024 if you want more training history.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Titles per encoding batch. Use 64 on CPU, 512+ on GPU. Default: 64.")
    p.add_argument("--threads", type=int, default=0,
                   help="CPU threads for PyTorch (0 = use all available cores). "
                        "Has no effect if running on GPU.")
    return p.parse_args()


def set_cpu_threads(n: int) -> int:
    """Set PyTorch CPU thread count. Returns the count actually set."""
    try:
        import torch
        count = n if n > 0 else os.cpu_count() or 4
        torch.set_num_threads(count)
        return count
    except ImportError:
        return n or (os.cpu_count() or 4)


def stream_patents(zip_path: pathlib.Path, year_start: int, year_end: int):
    """Yield (patent_id, patent_date, year, title) for each qualifying patent."""
    with zipfile.ZipFile(zip_path) as z:
        inner = z.namelist()[0]
        with z.open(inner) as raw:
            reader = csv.DictReader(
                io.TextIOWrapper(raw, encoding="utf-8", errors="replace"),
                delimiter="\t",
            )
            for row in reader:
                kind = (row.get("wipo_kind") or "").strip()
                if kind not in WIPO_GRANTED:
                    continue
                date_str = (row.get("patent_date") or "").strip()
                if len(date_str) < 4:
                    continue
                try:
                    year = int(date_str[:4])
                except ValueError:
                    continue
                if not (year_start <= year <= year_end):
                    continue
                title = (row.get("patent_title") or "").strip()
                if not title:
                    continue
                yield row.get("patent_id", ""), date_str, year, title


def embed_year(model, records: list, batch_size: int, year: int) -> np.ndarray:
    """Encode all titles for one year with a visible per-batch progress bar."""
    titles = [r[3] for r in records]
    return model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=True,   # tqdm bar showing batches + ETA
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


def main():
    args = parse_args()
    year_start, year_end = args.years
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_threads = set_cpu_threads(args.threads)
    print(f"CPU threads : {n_threads}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Year range  : {year_start}–{year_end}")
    print(f"Model       : {MODEL_ID}")

    # Check which years are already done
    already_done = {
        int(f.stem) for f in OUT_DIR.glob("*.npz")
        if f.stem.isdigit() and year_start <= int(f.stem) <= year_end
    }
    years_needed = [y for y in range(year_start, year_end + 1) if y not in already_done]
    if already_done:
        print(f"\nAlready embedded : {sorted(already_done)}")
    print(f"To embed         : {years_needed}")
    if not years_needed:
        print("\nAll years already embedded. Nothing to do.")
        return

    print(f"\nLoading model: {MODEL_ID} …")
    model = SentenceTransformer(MODEL_ID)
    print("Model loaded.\n")

    # ── Phase 1: Stream the TSV and bucket patents by year ────────────────────
    print(f"Streaming {PATENTS.name} …")
    buckets: dict[int, list] = defaultdict(list)
    total = 0
    t_read_start = time.time()

    with tqdm(desc="Reading patents", unit=" patents", unit_scale=True,
              mininterval=1, dynamic_ncols=True) as bar:
        for pid, date, year, title in stream_patents(PATENTS, year_start, year_end):
            if year not in already_done:
                buckets[year].append((pid, date, year, title))
            total += 1
            bar.update(1)
            if total % 200_000 == 0:
                bar.set_postfix_str(
                    f"kept={sum(len(v) for v in buckets.values()):,}  "
                    f"years={len(buckets)}"
                )

    t_read = time.time() - t_read_start
    kept = sum(len(v) for v in buckets.values())
    print(f"\nRead {total:,} rows in {fmt_time(t_read)} "
          f"— kept {kept:,} patents across {len(buckets)} years.\n")

    # ── Phase 2: Embed year by year ───────────────────────────────────────────
    speed_history: list[float] = []   # patents/sec per completed year

    for i, year in enumerate(sorted(buckets.keys())):
        out_path = OUT_DIR / f"{year}.npz"
        records  = buckets[year]
        n        = len(records)

        # ETA estimate based on previous years
        if speed_history:
            avg_speed = sum(speed_history) / len(speed_history)
            remaining_patents = sum(len(buckets[y]) for y in sorted(buckets)[i:])
            eta = remaining_patents / avg_speed
            eta_str = f"  (est. {fmt_time(eta)} remaining for {len(sorted(buckets))-i} years)"
        else:
            eta_str = ""

        print(f"── {year}  {n:,} patents{eta_str}")

        t0 = time.time()
        embeddings = embed_year(model, records, args.batch_size, year)
        elapsed    = time.time() - t0
        speed      = n / elapsed if elapsed > 0 else 0
        speed_history.append(speed)

        ids    = np.array([r[0] for r in records])
        dates  = np.array([r[1] for r in records])
        titles = np.array([r[3] for r in records])

        np.savez_compressed(out_path, ids=ids, dates=dates,
                            titles=titles, embeddings=embeddings)
        mb = out_path.stat().st_size / 1e6
        print(f"   saved {mb:.0f} MB → {out_path.name}  "
              f"[{fmt_time(elapsed)}, {speed:.0f} patents/sec]\n")

    years_done = sorted(int(f.stem) for f in OUT_DIR.glob("*.npz") if f.stem.isdigit())
    print(f"Done. {len(years_done)} year files in {OUT_DIR}/")
    print(f"Years: {years_done}")


if __name__ == "__main__":
    main()
