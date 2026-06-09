"""
add_assignees.py — "who's filing" per cluster

Joins each cluster's sample patents (cluster_titles_ids.pkl, ~25 patents/cluster)
to PatentsView's g_assignee_disambiguated.tsv to find the top filing
organizations per cluster.

Note: based on each cluster's REPRESENTATIVE sample patents (the full per-patent
cluster assignment wasn't persisted), so it's a sampled — not exhaustive — view.

Writes cluster_assignees.pkl:  {cluster_id: [(org_name, count), ...]}

Run:  python scripts/add_assignees.py
"""
import pickle, pathlib, zipfile, io, csv
from collections import Counter, defaultdict

BASE = pathlib.Path(__file__).parent.parent
CDIR = BASE / 'data' / 'processed' / 'clusters'
ASG  = BASE / 'data' / 'raw' / 'patents' / 'g_assignee_disambiguated.tsv.zip'
OUT  = CDIR / 'cluster_assignees.pkl'
TOP_N = 5


def main():
    titles_ids = pickle.load(open(CDIR / 'cluster_titles_ids.pkl', 'rb'))

    # patent_id -> list of cluster_ids it represents
    pid2clusters = defaultdict(list)
    for cid, items in titles_ids.items():
        for pid, _ in items:
            if pid:
                pid2clusters[str(pid)].append(int(cid))
    print(f'{len(titles_ids)} clusters · {len(pid2clusters)} sample patent numbers to look up')

    # Stream the assignee file once, capture org for wanted patent_ids
    pid2org = {}
    z = zipfile.ZipFile(ASG)
    name = [n for n in z.namelist() if n.endswith('.tsv')][0]
    scanned = 0
    with z.open(name) as f:
        reader = csv.reader(io.TextIOWrapper(f, encoding='utf-8', errors='replace'), delimiter='\t')
        header = next(reader)
        i_pid = header.index('patent_id')
        i_org = header.index('disambig_assignee_organization')
        for row in reader:
            scanned += 1
            if scanned % 2_000_000 == 0:
                print(f'  scanned {scanned:,} assignee rows · matched {len(pid2org):,}/{len(pid2clusters):,}')
            if len(row) <= i_org:
                continue
            pid = row[i_pid]
            if pid in pid2clusters and pid not in pid2org:
                org = row[i_org].strip()
                if org:
                    pid2org[pid] = org
                if len(pid2org) == len(pid2clusters):
                    break

    # title lookup per patent (from the sample titles)
    pid_title = {}
    for cid, items in titles_ids.items():
        for pid, title in items:
            if pid:
                pid_title[str(pid)] = title

    # Aggregate per cluster: org -> list of (patent_id, title)
    cluster_org_pats = defaultdict(lambda: defaultdict(list))
    for pid, clusters in pid2clusters.items():
        org = pid2org.get(pid)
        if not org:
            continue
        title = pid_title.get(pid, '')
        for cid in clusters:
            cluster_org_pats[cid][org].append((pid, title))

    # summary: [(org, count), ...]   (cluster_assignees.pkl — unchanged shape)
    enriched = {}
    # detailed: [(org, count, [(pid, title), ...]), ...]  (cluster_filer_patents.pkl)
    detailed = {}
    for cid in titles_ids:
        orgs = cluster_org_pats.get(int(cid), {})
        ranked = sorted(orgs.items(), key=lambda kv: len(kv[1]), reverse=True)
        enriched[int(cid)] = [(org, len(pats)) for org, pats in ranked[:TOP_N]]
        detailed[int(cid)] = [(org, len(pats), pats) for org, pats in ranked]

    pickle.dump(enriched, open(OUT, 'wb'))
    pickle.dump(detailed, open(CDIR / 'cluster_filer_patents.pkl', 'wb'))

    n_with = sum(1 for v in enriched.values() if v)
    print(f'\n{n_with}/{len(enriched)} clusters have at least one identified filer')
    print(f'Saved → {OUT}')
    print(f'Saved → {CDIR / "cluster_filer_patents.pkl"}')
    # show a few
    for cid in list(enriched)[:6]:
        orgs = enriched[cid]
        if orgs:
            print(f'  C{cid}: ' + ', '.join(f'{o} ({n})' for o, n in orgs[:3]))


if __name__ == '__main__':
    main()
