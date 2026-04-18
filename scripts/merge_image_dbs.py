"""
Merge multiple image_features.db files into one.

Usage on Colab:
    python scripts/merge_image_dbs.py \
        --dbs /path/to/image_features_comparis.db \
               /path/to/image_features_sred.db \
               /path/to/image_features_robinreal.db \
        --out /path/to/image_features_merged.db
"""

import argparse
import sqlite3
from pathlib import Path


def merge(db_paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Open output DB and copy schema from first source
    out = sqlite3.connect(out_path)
    out.execute("PRAGMA journal_mode=WAL")

    first = sqlite3.connect(db_paths[0])
    schema_rows = first.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL AND name != 'sqlite_sequence'"
    ).fetchall()
    first.close()

    for (sql,) in schema_rows:
        out.execute(sql)
    out.commit()

    total = 0
    for db_path in db_paths:
        if not db_path.exists():
            print(f"  SKIP (not found): {db_path}")
            continue

        src = sqlite3.connect(db_path)
        src.row_factory = sqlite3.Row

        tables = [r[0] for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
        ).fetchall()]

        for table in tables:
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                continue
            cols = rows[0].keys()
            placeholders = ",".join("?" * len(cols))
            col_list = ",".join(cols)
            inserted = 0
            for row in rows:
                try:
                    out.execute(
                        f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})",
                        tuple(row)
                    )
                    inserted += 1
                except Exception as e:
                    print(f"  Row error ({table}): {e}")
            out.commit()
            print(f"  {db_path.name} → {table}: {inserted}/{len(rows)} rows inserted")
            total += inserted
        src.close()

    # Summary
    for table in ["listing_images", "image_features"]:
        try:
            count = out.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"Merged {table}: {count} rows")
        except Exception:
            pass

    out.close()
    print(f"\nDone — {total} total rows written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbs", nargs="+", required=True, help="Source DB paths")
    parser.add_argument("--out", required=True, help="Output merged DB path")
    args = parser.parse_args()

    db_paths = [Path(p) for p in args.dbs]
    out_path = Path(args.out)

    print(f"Merging {len(db_paths)} databases → {out_path}\n")
    merge(db_paths, out_path)


if __name__ == "__main__":
    main()
