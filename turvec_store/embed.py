"""Embedder: ubah teks -> vektor float32 ternormalisasi.

Dua backend:
- HashingEmbedder : TANPA dependensi, deterministik, langsung jalan. Cocok utk
  eksperimen/uji infrastruktur. Kualitas semantik terbatas (bukan model neural).
- SentenceTransformerEmbedder : embedding semantik sungguhan (perlu
  `pip install sentence-transformers`). Pakai ini utk hasil nyata.

Keduanya punya method `.encode(texts) -> np.ndarray (n, dim) float32` dan atribut
`.dim`, jadi bisa ditukar bebas di VectorStore.
"""
from __future__ import annotations

import re
import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    """Bag-of-words hashing + signed-hash ke `dim` dimensi, lalu L2-normalize.

    Deterministik dan tanpa dependensi eksternal. Bukan pengganti embedding
    neural, tapi cukup untuk menguji pipeline TurboVec end-to-end secara offline.
    """

    name = "hashing"

    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    def _embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text):
            h = hash(tok)
            idx = (h & 0x7FFFFFFF) % self.dim
            sign = 1.0 if (h >> 31) & 1 else -1.0
            v[idx] += sign
        return v

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = self._embed_one(t)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    """Embedding semantik sungguhan via sentence-transformers (lokal, offline
    setelah model ter-download). Default model kecil 384-dim (~80MB)."""

    # default multibahasa (mendukung Bahasa Indonesia); ganti ke
    # "sentence-transformers/all-MiniLM-L6-v2" kalau korpus murni Inggris.
    DEFAULT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model: str = DEFAULT):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model)
        try:
            self.dim = self._model.get_embedding_dimension()
        except AttributeError:  # versi lama
            self.dim = self._model.get_sentence_embedding_dimension()
        self.name = f"st:{model}"

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return emb.astype(np.float32)


def make_embedder(kind: str = "hashing", **kw):
    """Factory: 'hashing' (default, no-deps) atau 'sentence-transformers'."""
    if kind in ("hashing", "hash"):
        return HashingEmbedder(**kw)
    if kind in ("sentence-transformers", "st", "minilm"):
        return SentenceTransformerEmbedder(**kw)
    raise ValueError(f"embedder tidak dikenal: {kind!r}")
