"""Helpers for the code and batch domains: partial reads of large Hub parquet files.

Some sources keep the few fields we need (a patch, a label) next to very large ones
(full agent trajectories), or are much larger than the sample we draw. Reading only the
needed columns (and, optionally, the first row groups) over HTTP range requests keeps the
download small; the extracted table is cached under .cache/downloads like fetch().
"""

from __future__ import annotations

from pathlib import Path

from ...env import load_env
from ..common import CACHE


def hub_columns(repo: str, files: list[str], columns: list[str], cache_name: str, row_groups: int | None = None):
    """Selected columns of one or more parquet files of a Hub dataset (all row groups, or the first
    `row_groups` of each file), as a pandas DataFrame."""
    import pandas as pd

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{cache_name}.parquet"
    if not path.exists():
        load_env()
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        frames = []
        for name in files:
            with fs.open(f"datasets/{repo}/{name}", block_size=8 * 2**20) as handle:
                if row_groups is None:
                    table = pq.read_table(handle, columns=columns, pre_buffer=True)
                else:
                    parquet = pq.ParquetFile(handle, pre_buffer=True)
                    groups = list(range(min(row_groups, parquet.metadata.num_row_groups)))
                    table = pa.concat_tables(parquet.read_row_group(g, columns=columns) for g in groups)
                frames.append(table.to_pandas())
        frame = pd.concat(frames, ignore_index=True)
        tmp = Path(str(path) + ".part")
        frame.to_parquet(tmp)
        tmp.rename(path)
    return pd.read_parquet(path)
