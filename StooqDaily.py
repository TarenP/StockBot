#!/usr/bin/env python3
"""
Stooq Loader – reliable bulk ingestion
======================================
Scans all Stooq daily OHLCV *.txt files under a root folder,
parses both headered and extended formats, and writes a single
Parquet panel ready for downstream joins with sentiment data.

Supported file formats:
1. Headered: "Date,Open,High,Low,Close,Volume,Adj Close"
2. Extended: "TICKER,PER,DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL,OPENINT"

Usage:
    python stooq_loader.py
or import `consolidate` and call it.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from joblib import Parallel, delayed

pd.options.mode.copy_on_write = True  # pandas 2.2+

###############################################################################
# CSV parsing
###############################################################################
def _read_stooq(path: Path) -> pd.DataFrame | None:
    """Parse a single Stooq file to a DataFrame with ['open','high','low','close','volume'] and a DatetimeIndex."""
    try:
        first = path.open().readline().strip()
    except Exception as e:
        sys.stderr.write(f"[ERR] Failed to open {path.name}: {e}\n")
        return None

    # Choose parsing strategy
    try:
        if first.lower().startswith('date,'):
            df = pd.read_csv(path)
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        else:
            names = ['ticker0','per','date','time','open','high','low','close','volume','openint']
            df = pd.read_csv(path, header=None, names=names)
    except Exception as e:
        sys.stderr.write(f"[ERR] Could not read {path.name}: {e}\n")
        return None

    # Keep only needed columns
    col_map = {
        'date': 'date',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'vol': 'volume'
    }
    cols = [k for k in col_map if k in df.columns]
    if 'date' not in cols:
        return None
    df = df[cols].rename(columns={k: col_map[k] for k in cols})

    # Parse date
    date_str = df['date'].astype(str)
    fmt = '%Y-%m-%d' if date_str.str.contains('-').any() else '%Y%m%d'
    parsed = pd.to_datetime(date_str, format=fmt, errors='coerce')
    df = df.loc[parsed.notna()].copy()
    if df.empty:
        return None
    df.index = parsed[parsed.notna()].dt.normalize()
    df.drop(columns=['date'], inplace=True)

    # Cast numerics and drop NaNs
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['open','high','low','close','volume'], inplace=True)

    return df

###############################################################################
# Wrapper
###############################################################################

def _load_one(task: Tuple[str, Path]) -> pd.DataFrame | None:
    ticker, path = task
    df = _read_stooq(path)
    if df is None:
        return None
    df['ticker'] = ticker
    return df

###############################################################################
# Public API
###############################################################################

def consolidate(root: Path, output: Path, n_workers: int = 8) -> None:
    """Scan root for .txt, parse in parallel, and write a Parquet panel."""
    paths = {p.stem.split('.')[0].upper(): p for p in root.rglob('*.txt')}
    print(f"[INFO] Found {len(paths):,} files under {root}.")

    tasks = list(paths.items())
    # Parse in parallel and collect results
    results: List[pd.DataFrame | None] = Parallel(
        n_jobs=n_workers, verbose=5, backend='loky'
    )(
        delayed(_load_one)(t) for t in tasks
    )
    frames = [df for df in results if df is not None]

    if not frames:
        sys.stderr.write("[ERR] No Stooq files could be parsed.\n")
        return

    panel = pd.concat(frames, ignore_index=False)
    panel.sort_index(inplace=True)
    panel.to_parquet(output, index=True)
    print(f"[OK] Wrote {output} with {panel['ticker'].nunique():,} tickers, {len(panel):,} rows.")

###############################################################################
# CLI entrypoint
###############################################################################
if __name__ == '__main__':
    ROOT_DIR = Path('data')
    OUTPUT = Path('stooq_panel.parquet')
    consolidate(ROOT_DIR, OUTPUT)
