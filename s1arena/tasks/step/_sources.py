"""Download helpers shared by the step-domain adapters (Google Drive folders, Hub files)."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from ...env import load_env
from ..common import CACHE, fetch

_ENTRY = re.compile(r'<a href="([^"]+)"[^>]*>.*?<div class="flip-entry-title">([^<]+)</div>', re.S)


def drive_folder(folder_id: str) -> list[tuple[str, str]]:
    """(name, id) of the entries of a public Google Drive folder, read from its embedded view once and cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"gdrive_folder_{folder_id}.json"
    if not path.exists():
        request = urllib.request.Request(f"https://drive.google.com/embeddedfolderview?id={folder_id}", headers={"User-Agent": "s1arena"})
        with urllib.request.urlopen(request, timeout=120) as response:
            html = response.read().decode("utf-8", errors="replace")
        entries = []
        for href, name in _ENTRY.findall(html):
            match = re.search(r"/(?:d|folders)/([\w-]+)", href)
            if match:
                entries.append((name.strip(), match.group(1)))
        if not entries:
            raise RuntimeError(f"empty or unreadable Google Drive folder {folder_id}")
        path.write_text(json.dumps(entries), encoding="utf-8")
    return [tuple(e) for e in json.loads(path.read_text(encoding="utf-8"))]


def drive_file(file_id: str, name: str) -> Path:
    """A small public Google Drive file (no virus-scan interstitial below 100 MB), downloaded once."""
    return fetch(f"https://drive.google.com/uc?export=download&id={file_id}", name)


def hub_file(repo: str, filename: str, revision: str | None = None) -> str:
    load_env()
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def hub_parquet_columns(repo: str, files: list[str], columns: list[str], cache_name: str) -> list[dict]:
    """Selected columns of Hub parquet files, read over HTTP range requests (skipping e.g. screenshot columns) and cached."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = CACHE / f"{cache_name}.parquet"
    if not path.exists():
        load_env()
        from huggingface_hub import HfFileSystem

        fs, tables = HfFileSystem(), []
        for name in files:
            with fs.open(f"datasets/{repo}/{name}", block_size=8 * 2**20) as handle:
                tables.append(pq.read_table(handle, columns=columns, pre_buffer=True))
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".part")
        pq.write_table(pa.concat_tables(tables), tmp)
        tmp.rename(path)
    return pq.read_table(path).to_pylist()


_ENCODER = None
FAST_CHARS = 45_000  # a JSON string this short cannot exceed 30k o200k tokens (at worst about 1.5 characters per token)


def token_len(text: str) -> int:
    """o200k tokens of the text as a JSON string, as the builder counts the serialized request."""
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("o200k_base")
    return len(_ENCODER.encode(json.dumps(text, ensure_ascii=False), disallowed_special=()))


def fits(*texts: str, limit: int | None = None) -> bool:
    """Whether the texts stay under a token limit (none by default: the benchmark has no shared input budget)."""
    if limit is None:  # no shared input budget any more
        return True
    if sum(len(json.dumps(t, ensure_ascii=False)) for t in texts) <= FAST_CHARS * limit / 30_000:
        return True
    total = 0
    for text in texts:
        total += token_len(text)
        if total > limit:
            return False
    return True


def longest_fitting(blocks: list[str], k: int, fixed: str = "", limit: int | None = None) -> int:
    """The largest j <= k such that the first j blocks (joined) plus the fixed text fit the budget; 0 if none."""
    if limit is None:
        return k
    if fits("\n\n".join(blocks[:k]), fixed, limit=limit):
        return k
    budget = limit - token_len(fixed)
    total, j = 0, 0
    for block in blocks[:k]:
        total += token_len(block) + 2
        if total > budget:
            break
        j += 1
    return j
