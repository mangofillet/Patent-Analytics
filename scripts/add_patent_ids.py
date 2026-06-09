"""
add_patent_ids.py — recover patent numbers for cluster sample titles

cluster_titles.pkl stores only title strings. This streams the source
g_patent.tsv (which has patent_id + patent_title) once, matches each sample
title back to its patent number, and writes cluster_titles_ids.pkl:

    {cluster_id: [(patent_id, title), ...], ...}

Patent numbers link to Google Patents:  https://patents.google.com/patent/US{id}

Run:  python scripts/add_patent_ids.py
"""
import pickle, pathlib, zipfile, io, csv

BASE = pathlib.Path(__file__).parent.parent
CDIR = BASE / 'data' / 'processed' / 'clusters'
TSV  = BASE / 'data' / 'raw' / 'patents' / 'g_patent.tsv.zip'
OUT  = CDIR / 'cluster_titles_ids.pkl'


def main():
    titles = pickle.load(open(CDIR / 'cluster_titles.pkl', 'rb'))

    # Collect every wanted title (normalised) → list of (cluster_id, original)
    wanted = {}
    for cid, ts in titles.items():
        for t in ts:
            key = str(t).strip().lower()
            wanted.setdefault(key, []).append((int(cid), str(t)))
    print(f'{len(titles)} clusters · {sum(len(v) for v in titles.values())} sample titles '
          f'· {len(wanted)} unique to look up')

    # Stream g_patent.tsv once, capture first patent_id per wanted title
    found = {}  # normalised title -> patent_id
    z = zipfile.ZipFile(TSV)
    name = [n for n in z.namelist() if n.endswith('.tsv')][0]
    scanned = 0
    with z.open(name) as f:
        reader = csv.reader(io.TextIOWrapper(f, encoding='utf-8'), delimiter='\t')
        header = next(reader)
        i_id, i_title = header.index('patent_id'), header.index('patent_title')
        for row in reader:
            scanned += 1
            if scanned % 1_000_000 == 0:
                print(f'  scanned {scanned:,} patents · matched {len(found):,}/{len(wanted):,}')
            if len(row) <= i_title:
                continue
            key = row[i_title].strip().lower()
            if key in wanted and key not in found:
                found[key] = row[i_id]
                if len(found) == len(wanted):
                    break

    # Rebuild per-cluster lists in original order, with ids where available
    enriched = {}
    for cid, ts in titles.items():
        out = []
        for t in ts:
            pid = found.get(str(t).strip().lower())
            out.append((pid, str(t)))
        enriched[int(cid)] = out

    pickle.dump(enriched, open(OUT, 'wb'))
    matched = sum(1 for v in enriched.values() for pid, _ in v if pid)
    total   = sum(len(v) for v in enriched.values())
    print(f'\nMatched {matched}/{total} titles to patent numbers')
    print(f'Saved → {OUT}')
    # show a few
    cid0 = next(iter(enriched))
    print(f'\nSample (cluster {cid0}):')
    for pid, t in enriched[cid0][:4]:
        print(f'  US{pid if pid else "?":<10} {t[:60]}')


if __name__ == '__main__':
    main()
