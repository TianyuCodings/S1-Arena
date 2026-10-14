#!/bin/bash
# WebShop data from the Hugging Face mirror YWZBrandon/webshop-data at a pinned revision.
set -e
REV=ce990fff5aee388db2706f07820c578ab68e0453
BASE=https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/$REV
cd $HOME/s1arena_env/webshop/data
for f in items_human_ins.json items_ins_v2_1000.json items_shuffle_1000.json subsets/10k/items_human_ins_10000.json subsets/10k/items_ins_v2_10000.json subsets/10k/items_shuffle_10000.json subsets/10k/documents_10000.jsonl items_ins_v2.json items_shuffle.json; do
  if [ ! -f "$f" ]; then
    curl -sSfL --retry 5 -o "$f.part" "$BASE/$f" && mv "$f.part" "$f"
    echo "$(date -u +%H:%M:%S) done $f $(stat -c %s $f)"
  fi
done
sha256sum items_shuffle.json items_ins_v2.json items_human_ins.json items_ins_v2_1000.json items_shuffle_1000.json subsets/10k/* > SHA256SUMS
echo ALL DONE
