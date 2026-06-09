"""Demo TurboVec: cek recall (vs brute-force exact) + kompresi memori + kecepatan."""
import time
import numpy as np
import turbovec

rng = np.random.default_rng(42)
N, DIM, K = 50_000, 1536, 10        # 50k dokumen, dimensi ala OpenAI embedding
BIT = 4                              # bit_width → kompresi 32/BIT = 8x (4-bit)
N_CLUSTERS = 200                     # struktur klaster spt embedding sungguhan

# Data berklaster + ternormalisasi (mirip embedding semantik, bukan noise murni)
centers = rng.standard_normal((N_CLUSTERS, DIM)).astype(np.float32)
labels = rng.integers(0, N_CLUSTERS, N)
data = centers[labels] + 0.35 * rng.standard_normal((N, DIM)).astype(np.float32)
data /= np.linalg.norm(data, axis=1, keepdims=True)
queries = data[:100]                 # pakai 100 vektor pertama sbg query

# --- Bangun index TurboVec ---
t = time.perf_counter()
idx = turbovec.TurboQuantIndex(dim=DIM, bit_width=BIT)
idx.add(data)
idx.prepare()
build_s = time.perf_counter() - t

# --- Search TurboVec ---
t = time.perf_counter()
approx = idx.search(queries, k=K)
tv_s = time.perf_counter() - t

# --- Brute-force exact (cosine = dot, krn sudah dinormalisasi) ---
t = time.perf_counter()
sims = queries @ data.T
exact = np.argpartition(-sims, K, axis=1)[:, :K]
bf_s = time.perf_counter() - t

# --- Recall@K: berapa % tetangga exact yang ketemu TurboVec ---
# search() -> (scores, ids); ambil ids
approx_ids = np.asarray(approx[1])
recall = np.mean([len(set(a) & set(e)) / K for a, e in zip(approx_ids, exact)])

# --- Memori ---
raw_mb = data.nbytes / 1e6
comp_mb = N * DIM * BIT / 8 / 1e6
print(f"Dataset     : {N:,} vektor x {DIM} dim, top-{K}")
print(f"Build+prep  : {build_s:.2f}s")
print(f"Search TV   : {tv_s*1000:.1f} ms (100 query)  |  brute-force: {bf_s*1000:.1f} ms")
print(f"Recall@{K}   : {recall*100:.1f}%")
print(f"Memori raw  : {raw_mb:.0f} MB (float32)")
print(f"Memori komp : {comp_mb:.0f} MB ({BIT}-bit)  →  {raw_mb/comp_mb:.1f}x lebih kecil")
