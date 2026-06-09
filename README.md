# Turvec — Pencarian Semantik Lokal di atas TurboVec

Mesin pencari **berdasarkan makna** (bukan sekadar kata yang sama) untuk koleksi
dokumen, berjalan **100% lokal** di komputer biasa. Dibangun di atas
[TurboVec](https://github.com/RyanCodrai/turbovec) yang memampatkan vektor
embedding hingga ~16×, sehingga jutaan dokumen tetap muat di RAM dan cepat dicari
tanpa server besar atau layanan cloud berbayar.

Contoh: mencari `"hewan peliharaan"` akan menemukan dokumen `"kucing oranye suka
tidur di sofa"` walaupun tidak ada satu kata pun yang sama.

---

## Daftar isi
1. [Cara kerja singkat](#cara-kerja-singkat)
2. [Struktur proyek & fungsi tiap file](#struktur-proyek--fungsi-tiap-file)
3. [Persiapan (sekali saja)](#persiapan-sekali-saja)
4. [Cara menjalankan — langkah demi langkah](#cara-menjalankan--langkah-demi-langkah)
   - [A. Lewat browser](#a-lewat-browser-paling-mudah)
   - [B. Lewat terminal (CLI)](#b-lewat-terminal-cli)
   - [C. Lewat kode Python](#c-lewat-kode-python)
   - [D. Simulasi 1 juta dokumen](#d-simulasi-1-juta-dokumen)
   - [E. Benchmark akurasi](#e-benchmark-akurasi)
5. [Angka hasil pengujian](#angka-hasil-pengujian)
6. [Catatan teknis penting](#catatan-teknis-penting)

---

## Cara kerja singkat

Ada tiga tahap. Turvec hanya bertanggung jawab atas tahap penyimpanan & pencarian;
pemahaman makna datang dari model embedding.

```
   TEKS   ──(1) embedding──▶  VEKTOR  ──(2) kompresi & simpan──▶  INDEX (TurboVec)
                                                                     │
   KUERI  ──(1) embedding──▶  VEKTOR  ──(3) cari termirip──────────▶ HASIL
```

| Tahap | Tugas | Pelaksana |
|-------|-------|-----------|
| (1) Embedding | ubah teks menjadi vektor angka yang mewakili maknanya | model `sentence-transformers` |
| (2) Kompresi & simpan | padatkan vektor ~16× lalu simpan | TurboVec |
| (3) Pencarian | temukan vektor paling mirip dengan kueri | TurboVec |

Poin penting: **kualitas hasil ditentukan oleh model embedding**, sedangkan
**kemampuan menampung data besar ditentukan oleh TurboVec**.

---

## Struktur proyek & fungsi tiap file

```
.
├── turvec_store/            # paket inti (library)
│   ├── __init__.py          # titik impor: VectorStore, Hit, make_embedder, ...
│   ├── embed.py             # ubah teks -> vektor (2 backend)
│   ├── store.py             # VectorStore: bungkus TurboVec + metadata + simpan/muat
│   └── scale.py             # ScaleStore: versi hemat-RAM untuk dataset sangat besar
├── cli.py                   # antarmuka terminal: add / search / info
├── webapp.py                # antarmuka browser (http://localhost:8000)
├── simulate.py              # pembuat dataset simulasi (default 1 juta dokumen)
├── demo_recall.py           # benchmark akurasi vs pencarian exact + rasio kompresi
├── requirements.txt         # daftar dependensi
├── README.md                # dokumen ini
└── data/                    # database tersimpan (dibuat saat dipakai)
```

### `turvec_store/embed.py` — Teks menjadi vektor
Menyediakan dua "embedder", keduanya punya method `encode(daftar_teks)` dan atribut `dim`:

- **`HashingEmbedder`** — tanpa dependensi tambahan, langsung jalan offline.
  Mengubah teks jadi vektor dengan teknik hashing kata. Cepat, tetapi hanya bisa
  mencocokkan **kata yang sama persis** (tidak paham makna). Cocok untuk uji cepat.
- **`SentenceTransformerEmbedder`** — embedding **semantik** sungguhan memakai
  model neural. Default model bersifat **multibahasa** (mendukung Bahasa
  Indonesia). Perlu paket `sentence-transformers`.
- **`make_embedder(kind)`** — pemilih praktis: `"hashing"` atau `"st"`.

### `turvec_store/store.py` — Inti yang menyatukan semuanya
TurboVec hanya menyimpan vektor + id angka; ia tidak menyimpan teks asli. Kelas
**`VectorStore`** menambal kekurangan itu:

- `add(daftar_teks)` — meng-embed teks lalu memasukkannya ke index, sambil
  menyimpan peta `id -> teks` di memori.
- `search(kueri, k)` — meng-embed kueri, mencari `k` termirip, mengembalikan
  daftar objek `Hit` berisi `id`, `score`, `text`, `meta`.
- `remove(id)`, `__len__()`.
- `save(path)` / `load(path)` — persistensi ke dua berkas berdampingan:
  `<nama>.tv` (index biner TurboVec) dan `<nama>.json` (peta teks + konfigurasi).

### `turvec_store/scale.py` — Untuk dataset sangat besar
**`ScaleStore`** dipakai saat dokumen berjumlah sangat banyak (mis. 1 juta).
Bedanya dengan `VectorStore`: teks disimpan di berkas `.txt` baris-per-dokumen
(nomor baris = id) sehingga hemat memori dan cepat dimuat, bukan dimuat sebagai
satu objek JSON besar. Bersifat **baca-saja** (fokus pencarian).

### `cli.py` — Pemakaian dari terminal
Tiga sub-perintah: `add`, `search`, `info`. Mendukung penambahan banyak teks
sekaligus atau dari sebuah berkas (`--file`, satu baris = satu dokumen), serta
pemilihan embedder lewat `--embedder`.

### `webapp.py` — Pemakaian dari browser
Server web ringan (memakai pustaka bawaan Python, tanpa framework). Menyajikan
satu halaman dengan kotak pencarian, kontrol jumlah hasil, skor relevansi, dan
waktu pencarian. Otomatis memakai dataset besar `data/scale.*` bila tersedia,
jika tidak memakai `data/web`.

### `simulate.py` — Membuat data uji berskala besar
Membangun N dokumen dummy (default 1.000.000). Agar tidak perlu menunggu lama,
hanya beberapa puluh kalimat template yang di-embed dengan model neural, lalu
jutaan dokumen dihasilkan sebagai variasi template. Hasil vektor tetap berklaster
secara semantik sehingga pencarian makna tetap berfungsi.

### `demo_recall.py` — Mengukur mutu kompresi
Membandingkan hasil pencarian TurboVec (vektor terkompresi) dengan pencarian
exact (brute-force), serta menghitung penghematan memori pada berbagai tingkat
kompresi.

---

## Persiapan (sekali saja)

Butuh Python 3 dan koneksi internet untuk unduhan awal.

```bash
# 1. Masuk ke folder proyek
cd Turvec

# 2. Buat lingkungan virtual
python -m venv .venv

# 3. Pasang dependensi inti (TurboVec + numpy)
.venv/bin/pip install -r requirements.txt

# 4. Pasang model embedding semantik (untuk pencarian berbasis makna)
.venv/bin/pip install sentence-transformers
```

> Catatan: langkah 4 mengunduh paket yang cukup besar. Tanpanya, program tetap
> berjalan memakai embedder `hashing` (hanya cocokkan kata, bukan makna).
>
> Bila lingkungan virtual sudah diaktifkan di terminal, perintah `.venv/bin/python`
> dan `.venv/bin/pip` boleh disingkat menjadi `python` dan `pip`.

---

## Cara menjalankan — langkah demi langkah

### A. Lewat browser (paling mudah)

```bash
python webapp.py
```

Tunggu hingga muncul baris `Siap. Buka http://localhost:8000`, lalu buka alamat
tersebut di browser.

- Ketik kueri di kotak pencarian, tekan Enter.
- Klik salah satu chip contoh untuk mencoba cepat.
- Atur jumlah hasil lewat tombol 5 / 10 / 20.

Menghentikan server: tekan `Ctrl+C` di terminal.

Bila muncul pesan `Address already in use`, berarti masih ada server lama yang
berjalan. Bersihkan lalu jalankan ulang:

```bash
pkill -f webapp.py
python webapp.py
```

### B. Lewat terminal (CLI)

```bash
# Menambah dokumen (semantik, memakai model neural)
python cli.py add --db data/catatan --embedder st "kucing oranye suka tidur di sofa" "resep nasi goreng pakai telur"

# Menambah dokumen dari sebuah berkas teks (satu baris = satu dokumen)
python cli.py add --db data/catatan --embedder st --file dokumenku.txt

# Mencari berdasarkan makna (model dibaca otomatis dari database)
python cli.py search --db data/catatan "makanan pedas" -k 5

# Melihat informasi database
python cli.py info --db data/catatan
```

Argumen yang tersedia: `--db` (lokasi database, tanpa ekstensi), `--embedder`
(`hashing` atau `st`), `--bit` (tingkat kompresi 2/3/4), `-k` (jumlah hasil).

### C. Lewat kode Python

```python
from turvec_store import VectorStore, make_embedder

# Membuat store dengan embedder semantik
store = VectorStore(make_embedder("st"), bit_width=4)
store.add([
    "robot lengan mengambil barang di gudang",
    "anjing menggonggong di taman pagi hari",
])
store.save("data/contoh")

# Memuat kembali dan mencari
store = VectorStore.load("data/contoh")
for hit in store.search("hewan peliharaan", k=3):
    print(round(hit.score, 3), hit.text)
```

### D. Simulasi 1 juta dokumen

```bash
# Membangun dataset besar (sekali jalan, sekitar satu menit)
python simulate.py 1000000

# Lalu jalankan antarmuka browser; otomatis memakai dataset ini
python webapp.py
```

Pencarian pertama agak lambat (beberapa detik) karena ada pemanasan satu kali;
pencarian berikutnya hanya puluhan milidetik.

### E. Benchmark akurasi

```bash
python demo_recall.py
```

Menampilkan recall (kecocokan dengan pencarian exact) dan rasio penghematan
memori pada tingkat kompresi yang dipakai.

---

## Angka hasil pengujian

Diukur pada satu dataset simulasi 1.000.000 dokumen (vektor 384 dimensi,
kompresi 4-bit) di sebuah laptop kelas menengah:

| Aspek | Hasil |
|-------|-------|
| Memori vektor mentah (float32) | ~1,5 GB |
| Memori setelah kompresi TurboVec | ~0,19 GB (sekitar 8–16× lebih kecil) |
| Ukuran index di disk | ~204 MB |
| Waktu membangun 1 juta dokumen | ~1 menit |
| Waktu pencarian per kueri | ~60 ms |
| Akurasi vs pencarian exact | hasil terkompresi setara dengan exact |

Tingkat kompresi dapat diatur lewat `bit_width`:

| bit_width | Kompresi | Catatan |
|-----------|----------|---------|
| 2 | 16× | paling hemat, recall paling rendah |
| 3 | 10× | seimbang |
| 4 | 8× | recall paling baik (default) |

---

## Catatan teknis penting

- **Pilihan model menentukan mutu hasil.** Untuk teks Bahasa Indonesia, gunakan
  model **multibahasa** (sudah menjadi default pada embedder `st`). Model yang
  hanya dilatih untuk Bahasa Inggris dapat memberi peringkat yang salah pada teks
  Indonesia.
- **Embedder `hashing` tidak memahami makna.** Ia hanya mencocokkan kata yang
  sama. Gunakan embedder `st` untuk pencarian semantik sungguhan.
- **`bit_width` hanya menerima nilai 2, 3, atau 4.**
- **Pemampatan tidak merusak hasil.** Pada pengujian, peringkat hasil dari index
  terkompresi sama dengan pencarian exact; yang menentukan kualitas adalah model
  embedding, bukan kompresinya.
- **Format penyimpanan.** `VectorStore` menyimpan dua berkas: `<nama>.tv` (index)
  dan `<nama>.json` (teks + konfigurasi). `ScaleStore` memakai `<nama>.tv`,
  `<nama>.txt`, dan `<nama>.meta.json`.

---

## Lisensi & atribusi

Proyek ini memakai pustaka pihak ketiga
[TurboVec](https://github.com/RyanCodrai/turbovec) dan
[sentence-transformers](https://www.sbert.net/). Patuhi lisensi masing-masing.
