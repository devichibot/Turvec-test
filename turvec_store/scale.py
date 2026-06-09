"""ScaleStore — store read-mostly untuk dataset besar (mis. 1 juta dokumen).

Beda dari VectorStore: teks disimpan di file .txt baris-per-dokumen (id = nomor
baris) supaya hemat memori & cepat dimuat, bukan dict JSON besar. Dipakai oleh
webapp bila file data/scale.* ada.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import turbovec

from .embed import make_embedder
from .store import Hit


class ScaleStore:
    def __init__(self, index, texts: list[str], embedder, bit_width: int):
        self._index = index
        self._texts = texts          # index 0 -> id 1
        self.embedder = embedder
        self.bit_width = bit_width
        self.dim = embedder.dim
        self._prepared = False

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        name = meta["embedder"]
        embedder = make_embedder("st", model=name[3:]) if name.startswith("st:") \
            else make_embedder("hashing", dim=meta["dim"])
        t = time.perf_counter()
        index = turbovec.IdMapIndex.load(str(path.with_suffix(".tv")))
        texts = path.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
        print(f"ScaleStore: {len(texts):,} dokumen dimuat ({time.perf_counter()-t:.1f}s)")
        return cls(index, texts, embedder, meta["bit_width"])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self._prepared:
            t = time.perf_counter()
            self._index.prepare()
            self._prepared = True
            print(f"ScaleStore: prepare {time.perf_counter()-t:.1f}s (sekali saja)")
        qv = self.embedder.encode([query])
        scores, ids = self._index.search(qv, k=k)
        scores = np.asarray(scores)[0]
        ids = np.asarray(ids)[0]
        out = []
        for s, i in zip(scores.tolist(), ids.tolist()):
            i = int(i)
            txt = self._texts[i - 1] if 1 <= i <= len(self._texts) else ""
            out.append(Hit(i, float(s), txt, {}))
        return out

    def __len__(self) -> int:
        return len(self._index)
