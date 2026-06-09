#!/usr/bin/env bash
# Re-download the USPTO bulk patent file (excluded from git due to 222MB size).
# Source: PatentsView — https://patentsview.org/download/data-download-tables
set -e

DEST="data/raw/patents/g_patent.tsv.zip"

if [ -f "$DEST" ]; then
  echo "Already present: $DEST"
  exit 0
fi

mkdir -p data/raw/patents
echo "Downloading g_patent.tsv.zip from PatentsView..."
curl -L "https://s3.amazonaws.com/data.patentsview.org/download/g_patent.tsv.zip" -o "$DEST"
echo "Done: $DEST"
