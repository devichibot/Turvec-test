#!/usr/bin/env python
"""CLI TurboVec store.

Contoh:
  python cli.py add  --db data/notes  "teks dokumen pertama"  "dokumen kedua"
  python cli.py add  --db data/notes  --file catatan.txt        # 1 baris = 1 dok
  python cli.py search --db data/notes "kata kunci" -k 5
  python cli.py info --db data/notes

--embedder hashing (default, tanpa deps) | st (sentence-transformers, semantik nyata)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from turvec_store import VectorStore, make_embedder


def _open(db: str, embedder_kind: str, dim: int, bit: int) -> tuple[VectorStore, Path]:
    db_path = Path(db)
    if db_path.with_suffix(".json").exists():
        return VectorStore.load(db_path), db_path
    emb = make_embedder(embedder_kind, dim=dim) if embedder_kind in ("hashing", "hash") \
        else make_embedder(embedder_kind)
    return VectorStore(emb, bit_width=bit), db_path


def main() -> None:
    p = argparse.ArgumentParser(description="TurboVec local vector store")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", required=True, help="prefix file db (tanpa ekstensi)")
    common.add_argument("--embedder", default="hashing", help="hashing|st")
    common.add_argument("--dim", type=int, default=256, help="dim (embedder hashing)")
    common.add_argument("--bit", type=int, default=4, help="bit_width 2|3|4")

    a = sub.add_parser("add", parents=[common])
    a.add_argument("texts", nargs="*", help="teks dokumen")
    a.add_argument("--file", help="file teks, 1 baris = 1 dokumen")

    s = sub.add_parser("search", parents=[common])
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)

    sub.add_parser("info", parents=[common])

    args = p.parse_args()
    store, db_path = _open(args.db, args.embedder, args.dim, args.bit)

    if args.cmd == "add":
        texts = list(args.texts)
        if args.file:
            texts += [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not texts:
            p.error("tidak ada teks untuk ditambahkan")
        ids = store.add(texts)
        store.save(db_path)
        print(f"+ {len(ids)} dokumen ditambah (id {ids[0]}..{ids[-1]}). Total: {len(store)}")

    elif args.cmd == "search":
        for h in store.search(args.query, k=args.k):
            print(f"[{h.score:7.4f}] #{h.id}  {h.text[:90]}")

    elif args.cmd == "info":
        print(f"db        : {db_path}")
        print(f"embedder  : {getattr(store.embedder, 'name', '?')}")
        print(f"dim       : {store.dim}")
        print(f"bit_width : {store.bit_width}")
        print(f"dokumen   : {len(store)}")


if __name__ == "__main__":
    main()
