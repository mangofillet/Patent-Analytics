"""
build_patent_db.py — fast patent_id → title lookup for the dashboard's
"search any patent number" fallback.

Streams g_patent.tsv once and writes an indexed SQLite DB so the dashboard can
look up any patent's title instantly (then embed it + match to the nearest
cluster centroid for patents not in the sampled index).

One-time build (~2-3 min). Run:  python scripts/build_patent_db.py
"""
import sqlite3, zipfile, io, csv, pathlib

BASE = pathlib.Path(__file__).parent.parent
TSV  = BASE / 'data' / 'raw' / 'patents' / 'g_patent.tsv.zip'
DB   = BASE / 'data' / 'processed' / 'patent_titles.db'


def main():
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.execute('PRAGMA journal_mode=OFF')
    con.execute('PRAGMA synchronous=OFF')
    con.execute('CREATE TABLE patents (id TEXT PRIMARY KEY, title TEXT)')

    z = zipfile.ZipFile(TSV)
    name = [n for n in z.namelist() if n.endswith('.tsv')][0]
    batch, n = [], 0
    with z.open(name) as f:
        reader = csv.reader(io.TextIOWrapper(f, encoding='utf-8', errors='replace'), delimiter='\t')
        header = next(reader)
        i_id, i_title = header.index('patent_id'), header.index('patent_title')
        for row in reader:
            if len(row) <= i_title:
                continue
            batch.append((row[i_id], row[i_title]))
            if len(batch) >= 50000:
                con.executemany('INSERT OR IGNORE INTO patents VALUES (?,?)', batch)
                n += len(batch); batch = []
                if n % 1_000_000 == 0:
                    print(f'  inserted {n:,}')
    if batch:
        con.executemany('INSERT OR IGNORE INTO patents VALUES (?,?)', batch); n += len(batch)
    con.commit()
    con.close()
    print(f'Done — {n:,} patents indexed → {DB}  ({DB.stat().st_size/1e6:.0f} MB)')


if __name__ == '__main__':
    main()
