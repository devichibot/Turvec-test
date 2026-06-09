"""VectorStore: pembungkus TurboVec siap-pakai untuk laptop ini.

Menyatukan tiga hal yang dibutuhkan vector search nyata:
  1. embedder   : teks -> vektor
  2. TurboVec   : IdMapIndex (vektor terkompresi + id uint64)
  3. metadata   : peta id -> {text, meta}  (TurboVec tak menyimpan teks)

Persistensi = 2 file berdampingan:
  <name>.tv    -> index TurboVec (biner)
  <name>.json  -> metadata + konfigurasi (embedder, dim, bit_width, next_id)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import turbovec

from .embed import make_embedder


@dataclass
class Hit:
    id: int
    score: float
    text: str
    meta: dict


class VectorStore:
    def __init__(self, embedder, bit_width: int = 4):
        self.embedder = embedder
        self.bit_width = int(bit_width)
        self.dim = embedder.dim
        self._index = turbovec.IdMapIndex(dim=self.dim, bit_width=self.bit_width)
        self._meta: dict[int, dict] = {}   # id -> {"text":..., "meta":...}
        self._next_id = 1
        self._prepared = False

    # ---- tulis ----
    def add(self, texts: list[str], metas: list[dict] | None = None) -> list[int]:
        """Embed lalu masukkan teks ke index. Kembalikan id yang diberikan."""
        if isinstance(texts, str):
            texts = [texts]
        metas = metas or [{} for _ in texts]
        if len(metas) != len(texts):
            raise ValueError("jumlah metas harus sama dengan texts")

        vecs = self.embedder.encode(texts)
        ids = np.arange(self._next_id, self._next_id + len(texts), dtype=np.uint64)
        self._index.add_with_ids(vecs, ids)
        for i, _id in enumerate(ids.tolist()):
            self._meta[_id] = {"text": texts[i], "meta": metas[i]}
        self._next_id += len(texts)
        self._prepared = False
        return ids.tolist()

    def remove(self, _id: int) -> None:
        self._index.remove(int(_id))
        self._meta.pop(int(_id), None)
        self._prepared = False

    # ---- baca ----
    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self._prepared:
            self._index.prepare()
            self._prepared = True
        qv = self.embedder.encode([query])
        scores, ids = self._index.search(qv, k=k)
        scores = np.asarray(scores)[0]
        ids = np.asarray(ids)[0]
        hits = []
        for s, i in zip(scores.tolist(), ids.tolist()):
            rec = self._meta.get(int(i), {})
            hits.append(Hit(int(i), float(s), rec.get("text", ""), rec.get("meta", {})))
        return hits

    def __len__(self) -> int:
        return len(self._index)

    # ---- persistensi ----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        self._index.write(str(path.with_suffix(".tv")))
        sidecar = {
            "embedder": getattr(self.embedder, "name", "hashing"),
            "dim": self.dim,
            "bit_width": self.bit_width,
            "next_id": self._next_id,
            "meta": {str(k): v for k, v in self._meta.items()},
        }
        path.with_suffix(".json").write_text(
            json.dumps(sidecar, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path, embedder=None) -> "VectorStore":
        path = Path(path)
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if embedder is None:
            # rekonstruksi embedder hashing dgn dim yg sama; utk st: oper manual
            name = sidecar.get("embedder", "hashing")
            if name.startswith("st:"):
                embedder = make_embedder("st", model=name[3:])
            else:
                embedder = make_embedder("hashing", dim=sidecar["dim"])
        self = cls(embedder, bit_width=sidecar["bit_width"])
        self._index = turbovec.IdMapIndex.load(str(path.with_suffix(".tv")))
        self._meta = {int(k): v for k, v in sidecar["meta"].items()}
        self._next_id = sidecar["next_id"]
        return self
