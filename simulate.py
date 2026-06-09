#!/usr/bin/env python
"""Bangun database simulasi 1 JUTA dokumen dummy untuk diuji di browser.

Trik agar cepat (tanpa nunggu ~1 jam embedding):
  - Embed hanya ~puluhan KALIMAT TEMPLATE sungguhan dengan model neural (detik).
  - Hasilkan 1 juta dokumen sebagai variasi template + sedikit noise acak.
    Vektornya tetap BERKLASTER secara semantik nyata (pencarian makna jalan),
    tapi pembuatannya hitungan menit, bukan sejam.

Jalankan:
    python simulate.py            # default 1.000.000 dokumen
    python simulate.py 200000     # jumlah lain
Output: data/scale.tv (index TurboVec) + data/scale.txt + data/scale.meta.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

DB = Path("data/scale")
DIM_BIT = 4

# ---- kategori dummy: tiap entri = (template, kata-makna-untuk-embedding) ----
# {nama},{kota},{n} akan diisi nilai acak. Embedding pakai versi terisi contoh.
CATS = [
    "Pelanggan {nama} komplain paket belum sampai sudah {n} hari, minta pelacakan",
    "Transaksi {nama} gagal saat pembayaran kartu kredit, saldo terpotong",
    "{nama} melaporkan barang elektronik diterima dalam kondisi rusak pecah",
    "Permintaan pengembalian dana refund dari {nama} untuk pesanan nomor {n}",
    "Pesanan robot lengan industri otomasi untuk gudang oleh PT {nama}",
    "{nama} memesan paket katering nasi goreng dan ayam untuk {n} orang",
    "{nama} membeli makanan kucing dan anjing serta vitamin di toko hewan",
    "{nama} menyewa apartemen di kota {kota} selama {n} bulan",
    "{nama} membuat janji konsultasi dokter spesialis di klinik {kota}",
    "{nama} mendaftar kursus pemrograman python dan data science online",
    "{nama} servis mobil ganti oli dan rem di bengkel resmi {kota}",
    "{nama} mengajukan pinjaman kredit dan investasi emas senilai {n} juta",
    "Keluhan {nama} soal sinyal internet wifi lambat dan sering putus",
    "{nama} memesan tiket pesawat ke {kota} untuk liburan keluarga",
    "{nama} memesan kamar hotel bintang lima di {kota} dua malam",
    "{nama} membeli laptop gaming spesifikasi tinggi untuk desain grafis",
    "{nama} berlangganan layanan streaming film dan musik bulanan",
    "Pengaduan {nama} tentang tagihan listrik membengkak bulan ini",
    "{nama} memesan bunga dan kue ulang tahun untuk dikirim ke {kota}",
    "{nama} mendaftar keanggotaan gym dan kelas yoga di pusat kebugaran",
    "{nama} membeli pupuk dan bibit tanaman untuk kebun sayur organik",
    "{nama} memesan jasa pindahan rumah dan packing barang ke {kota}",
    "{nama} melaporkan akun diretas dan minta reset kata sandi keamanan",
    "{nama} membeli perlengkapan bayi popok dan susu formula",
    "{nama} memesan seragam batik kantor untuk {n} karyawan",
    "Pertanyaan {nama} soal garansi dan suku cadang mesin cuci",
    "{nama} memesan obat resep dokter dikirim ke alamat rumah",
    "{nama} mendaftar asuransi kesehatan keluarga premi bulanan",
    "{nama} membeli alat olahraga sepeda dan treadmill untuk di rumah",
    "{nama} memesan jasa fotografer pernikahan di gedung {kota}",
]

NAMA = ["Budi", "Siti", "Andi", "Rina", "Dewi", "Joko", "Maya", "Agus", "Putri",
        "Hadi", "Lestari", "Bayu", "Citra", "Eko", "Fitri", "Gita", "Indra",
        "Kartika", "Lukman", "Nadia", "Oki", "Wati", "Yusuf", "Zaki", "Tono"]
KOTA = ["Jakarta", "Surabaya", "Bandung", "Medan", "Semarang", "Makassar",
        "Yogyakarta", "Denpasar", "Palembang", "Balikpapan", "Malang", "Bogor"]


def fill(template: str, rng: np.random.Generator) -> str:
    return template.format(
        nama=NAMA[rng.integers(len(NAMA))],
        kota=KOTA[rng.integers(len(KOTA))],
        n=int(rng.integers(1, 99)),
    )


def main() -> None:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    DB.parent.mkdir(exist_ok=True)
    import turbovec
    import warnings
    warnings.filterwarnings("ignore")
    from turvec_store.embed import SentenceTransformerEmbedder

    print(f"== Simulasi {N:,} dokumen dummy ==")
    emb = SentenceTransformerEmbedder()
    DIM = emb.dim

    print("1/4  Embedding kalimat template (detik)...")
    t = time.perf_counter()
    centers = emb.encode([fill(c, np.random.default_rng(i)) for i, c in enumerate(CATS)])
    print(f"     {len(CATS)} template ter-embed dalam {time.perf_counter()-t:.1f}s")

    print(f"2/4  Membuat {N:,} vektor berklaster...")
    t = time.perf_counter()
    rng = np.random.default_rng(42)
    assign = rng.integers(0, len(CATS), N)
    vecs = centers[assign] + 0.18 * rng.standard_normal((N, DIM)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    print(f"     selesai {time.perf_counter()-t:.1f}s ({vecs.nbytes/1e9:.2f} GB sementara)")

    print(f"3/4  Menghasilkan teks & memasukkan ke TurboVec...")
    t = time.perf_counter()
    idx = turbovec.IdMapIndex(dim=DIM, bit_width=DIM_BIT)
    idx.add_with_ids(vecs, np.arange(1, N + 1, dtype=np.uint64))
    # teks ditulis langsung ke file (id = nomor baris, mulai 1)
    with open(DB.with_suffix(".txt"), "w", encoding="utf-8") as f:
        for a in assign:
            f.write(fill(CATS[a], rng) + "\n")
    print(f"     selesai {time.perf_counter()-t:.1f}s")

    print("4/4  Menyimpan index + metadata...")
    idx.write(str(DB.with_suffix(".tv")))
    DB.with_suffix(".meta.json").write_text(json.dumps({
        "embedder": emb.name, "dim": DIM, "bit_width": DIM_BIT, "count": N,
    }))
    tv_mb = DB.with_suffix(".tv").stat().st_size / 1e6
    print(f"\nSELESAI. {N:,} dokumen.")
    print(f"  index TurboVec di disk : {tv_mb:.0f} MB  (vs ~{N*DIM*4/1e6:.0f} MB float32 mentah)")
    print("  Jalankan:  python webapp.py   lalu buka http://localhost:8000")


if __name__ == "__main__":
    main()
