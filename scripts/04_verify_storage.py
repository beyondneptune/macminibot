"""Round-trip verification of the ArcticDB store after a backfill.

Lists every library + symbol, prints date range + row count + storage size on disk.
No IBKR connection. Safe to run anytime; ArcticDB LMDB supports concurrent readers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _storage import ARCTIC_PATH, open_store  # noqa: E402


def _dir_size_bytes(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    store = open_store()
    libs = sorted(store.list_libraries())
    print(f"ArcticDB store: {ARCTIC_PATH}")
    print(f"on-disk size:   {_fmt_size(_dir_size_bytes(ARCTIC_PATH))}")
    print(f"libraries:      {len(libs)}")
    print()
    total_rows = 0
    for lib_name in libs:
        lib = store[lib_name]
        syms = sorted(lib.list_symbols())
        print(f"== {lib_name} ({len(syms)} symbols) ==")
        for sym in syms:
            vit = lib.read(sym)
            df = vit.data
            md = vit.metadata or {}
            n = len(df)
            total_rows += n
            first = df.index.min() if n else "-"
            last = df.index.max() if n else "-"
            print(
                f"  {sym:25s}  {n:6d} rows  {first} → {last}  "
                f"({md.get('what_to_show', '?')})"
            )
        print()
    print(f"total rows across all libraries: {total_rows:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
