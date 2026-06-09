#!/usr/bin/env python
"""Tampilan browser untuk TurboVec store.

Jalankan:
    python webapp.py
lalu buka  http://localhost:8000  di browser.

Tanpa dependensi tambahan (pakai http.server bawaan Python).
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from turvec_store import VectorStore, make_embedder
from turvec_store.scale import ScaleStore

PORT = 8000

# ---- pilih store: dataset besar (data/scale.*) bila ada, jika tidak data/web ----
print("Memuat model embedding (sekali saja, mohon tunggu)...")
_scale = Path("data/scale")
_web = Path("data/web")
if _scale.with_suffix(".meta.json").exists():
    STORE = ScaleStore.load(_scale)
    SCALE_MODE = True
    DB = str(_scale)
    print(f"Mode SKALA: {len(STORE):,} dokumen.")
elif _web.with_suffix(".json").exists():
    STORE = VectorStore.load(_web)
    SCALE_MODE = False
    DB = str(_web)
    print(f"DB dimuat: {len(STORE)} dokumen.")
else:
    STORE = VectorStore(make_embedder("st"), bit_width=4)
    SCALE_MODE = False
    DB = str(_web)
    print("DB baru dibuat (kosong).")
print(f"Siap. Buka http://localhost:{PORT}")


PAGE = """<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Turvec</title>
<style>
 :root{
   --bg:#0d0e11; --surface:#15171c; --surface-2:#1a1d24; --line:#262a33;
   --txt:#e7e9ee; --mut:#868d9c; --faint:#5b6271;
   --acc:#e0a83d; --acc-soft:rgba(224,168,61,.14);
   --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,monospace;
   --radius:12px;
 }
 *{box-sizing:border-box}
 html,body{margin:0}
 body{background:var(--bg);color:var(--txt);
   font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
   -webkit-font-smoothing:antialiased}
 .wrap{max-width:760px;margin:0 auto;padding:48px 20px 80px}

 /* header */
 header{display:flex;align-items:baseline;justify-content:space-between;
   gap:16px;margin-bottom:28px}
 .brand{font-size:19px;font-weight:600;letter-spacing:-.01em}
 .brand b{color:var(--acc);font-weight:600}
 .meta{font:12px/1.4 var(--mono);color:var(--faint);text-align:right;white-space:nowrap}
 .meta .n{color:var(--mut)}

 /* search field */
 .field{position:relative;display:flex;align-items:center;
   background:var(--surface);border:1px solid var(--line);border-radius:14px;
   padding:4px 4px 4px 16px;transition:border-color .18s,box-shadow .18s}
 .field:focus-within{border-color:var(--acc);
   box-shadow:0 0 0 3px var(--acc-soft)}
 .field svg{flex:none;color:var(--mut)}
 #q{flex:1;background:none;border:0;outline:none;color:var(--txt);
   font:16px/1.2 inherit;padding:13px 12px}
 #q::placeholder{color:var(--faint)}
 #go{flex:none;background:var(--acc);color:#1a1408;border:0;border-radius:10px;
   padding:11px 20px;font:600 14px inherit;cursor:pointer;
   transition:transform .1s,filter .15s}
 #go:hover{filter:brightness(1.06)}
 #go:active{transform:translateY(1px)}
 #go:disabled{opacity:.55;cursor:wait}

 /* controls row */
 .controls{display:flex;align-items:center;gap:8px;margin:14px 2px 0;
   color:var(--mut);font-size:13px}
 .controls .lbl{color:var(--faint)}
 .seg{display:inline-flex;background:var(--surface);border:1px solid var(--line);
   border-radius:9px;overflow:hidden}
 .seg button{background:none;border:0;color:var(--mut);font:13px/1 var(--mono);
   padding:7px 11px;cursor:pointer}
 .seg button[aria-pressed=true]{background:var(--acc-soft);color:var(--acc)}
 .spent{margin-left:auto;font:12px/1 var(--mono);color:var(--faint)}

 /* example chips */
 .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
 .chip{background:var(--surface);border:1px solid var(--line);color:var(--mut);
   border-radius:999px;padding:7px 13px;font-size:13px;cursor:pointer;
   transition:border-color .15s,color .15s}
 .chip:hover{border-color:var(--acc);color:var(--txt)}

 /* results */
 #out{margin-top:26px}
 .res{border-top:1px solid var(--line)}
 .hit{display:grid;grid-template-columns:34px 1fr auto;gap:14px;align-items:center;
   padding:15px 4px;border-bottom:1px solid var(--line)}
 .rank{font:12px/1 var(--mono);color:var(--faint);text-align:right}
 .htext{min-width:0}
 .htext .t{overflow:hidden;text-overflow:ellipsis}
 .htext .id{font:11px/1.3 var(--mono);color:var(--faint);margin-top:3px}
 .sc{display:flex;flex-direction:column;align-items:flex-end;gap:5px;min-width:74px}
 .sc .v{font:600 13px/1 var(--mono);color:var(--acc);font-variant-numeric:tabular-nums}
 .sc .bar{height:3px;border-radius:2px;background:var(--acc);opacity:.85}

 .hint,.empty,.err{color:var(--mut);font-size:14px;padding:24px 4px;text-align:center}
 .err{color:#e07a5f}
 .empty b{color:var(--txt)}

 /* skeleton */
 .sk{display:grid;grid-template-columns:34px 1fr 74px;gap:14px;align-items:center;
   padding:15px 4px;border-bottom:1px solid var(--line)}
 .sk span{height:13px;border-radius:4px;background:linear-gradient(90deg,
   var(--surface),var(--surface-2),var(--surface));background-size:200% 100%;
   animation:sh 1.2s ease-in-out infinite}
 .sk .a{width:60%} .sk .b{justify-self:end;width:46px}
 @keyframes sh{0%{background-position:200% 0}100%{background-position:-200% 0}}

 .reveal{animation:rise .42s cubic-bezier(.16,1,.3,1) both}
 @keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
 @media (prefers-reduced-motion:reduce){.reveal{animation:none}.sk span{animation:none}}

 @media (max-width:560px){
   .wrap{padding:32px 16px 64px}
   header{flex-direction:column;gap:6px}.meta{text-align:left}
   #go{padding:11px 16px}
 }
</style></head><body><div class="wrap">

 <header>
   <div class="brand">Tur<b>vec</b></div>
   <div class="meta"><span class="n" id="count">&hellip;</span> dokumen terindeks<br id="modeline"></div>
 </header>

 <div class="field">
   <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
     stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.2-3.2"/></svg>
   <input id="q" placeholder="cari berdasarkan makna, mis. paket belum sampai" autofocus
     autocomplete="off" onkeydown="if(event.key==='Enter')doSearch()">
   <button id="go" onclick="doSearch()">Cari</button>
 </div>

 <div class="controls">
   <span class="lbl">Tampilkan</span>
   <div class="seg" id="kseg">
     <button data-k="5" aria-pressed="true">5</button>
     <button data-k="10" aria-pressed="false">10</button>
     <button data-k="20" aria-pressed="false">20</button>
   </div>
   <span class="spent" id="spent"></span>
 </div>

 <div class="chips" id="chips"></div>
 <div id="out"><div class="hint">Ketik kueri lalu tekan Enter.</div></div>
</div>

<script>
const EXAMPLES = ["paket belum sampai","pembayaran kartu kredit gagal",
  "pakan hewan peliharaan","pinjam dana dan beli emas","wifi lemot","servis kendaraan"];
let K = 5;

const $ = s => document.querySelector(s);
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

async function refresh(){
  try{
    const j = await (await fetch('/api/info')).json();
    $('#count').textContent = j.count.toLocaleString('id-ID');
    $('#modeline').insertAdjacentText && ($('#modeline').nextSibling);
    $('.meta').innerHTML = `<span class="n">${j.count.toLocaleString('id-ID')}</span> dokumen terindeks`
      + `<br>${esc(j.embedder)}`;
    $('#chips').innerHTML = EXAMPLES.map(e=>`<button class="chip" onclick="run(this.textContent)">${e}</button>`).join('');
  }catch(e){ $('.meta').textContent = 'gagal memuat info'; }
}
function run(q){ $('#q').value = q; doSearch(); }

document.getElementById('kseg').addEventListener('click', e=>{
  const b = e.target.closest('button'); if(!b) return;
  K = +b.dataset.k;
  [...e.currentTarget.children].forEach(x=>x.setAttribute('aria-pressed', x===b));
  if($('#q').value.trim()) doSearch();
});

function skeleton(){
  $('#out').innerHTML = '<div class="res">' + Array.from({length:Math.min(K,6)},()=>
    '<div class="sk"><span style="width:14px"></span><span class="a"></span><span class="b"></span></div>'
  ).join('') + '</div>';
}

async function doSearch(){
  const q = $('#q').value.trim(); if(!q) return;
  const go = $('#go'); go.disabled = true; skeleton();
  try{
    const j = await (await fetch('/api/search',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query:q,k:K})})).json();
    go.disabled = false;
    if(j.error){ $('#out').innerHTML = `<div class="err">${esc(j.error)}</div>`; return; }
    $('#spent').textContent = `${j.ms} ms`;
    if(!j.hits.length){
      $('#out').innerHTML = '<div class="empty"><b>Tidak ada hasil.</b><br>Index mungkin masih kosong.</div>';
      return;
    }
    const top = Math.max(...j.hits.map(h=>h.score), 0.0001);
    $('#out').innerHTML = '<div class="res">' + j.hits.map((h,i)=>{
      const w = Math.max(6, Math.round(h.score/top*100));
      return `<div class="hit reveal" style="animation-delay:${i*38}ms">
        <div class="rank">${i+1}</div>
        <div class="htext"><div class="t">${esc(h.text)}</div><div class="id">#${h.id}</div></div>
        <div class="sc"><span class="v">${h.score.toFixed(3)}</span>
          <span class="bar" style="width:${w}px"></span></div>
      </div>`;
    }).join('') + '</div>';
  }catch(e){
    go.disabled = false;
    $('#out').innerHTML = `<div class="err">Gagal menghubungi server.</div>`;
  }
}
refresh();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # senyapkan log per-request
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/info":
            self._json({
                "count": len(STORE),
                "scale": SCALE_MODE,
                "embedder": getattr(STORE.embedder, "name", "?"),
            })
        else:
            self.send_error(404)

    def do_POST(self):
        try:
            data = self._read()
            if self.path == "/api/add":
                if SCALE_MODE:
                    self._json({"error": "Mode dataset besar bersifat read-only (cari saja)."}, code=400)
                    return
                texts = [t for t in data.get("texts", []) if t.strip()]
                STORE.add(texts)
                STORE.save(DB)
                self._json({"added": len(texts), "total": len(STORE)})
            elif self.path == "/api/search":
                import time as _t
                t0 = _t.perf_counter()
                hits = STORE.search(data.get("query", ""), k=int(data.get("k", 5)))
                ms = (_t.perf_counter() - t0) * 1000
                self._json({
                    "ms": round(ms, 1),
                    "total": len(STORE),
                    "hits": [{"id": h.id, "score": h.score, "text": h.text} for h in hits],
                })
            else:
                self.send_error(404)
        except Exception as e:  # tampilkan error sbg JSON, jangan crash server
            self._json({"error": str(e)}, code=500)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
