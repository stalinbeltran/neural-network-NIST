"""App web LOCAL para visualizar la red competitiva en accion.

Para cada entrada del dataset muestra LADO A LADO dos grupos, con la MISMA red:
  - Original: la entrada 28x28 y el mapa 50x50 de neuronas activas (la ganadora en rojo).
  - Negativo: la MISMA entrada en negativo fotografico (1 - x) y su mapa de neuronas activas.
Ademas, el campo receptivo de la neurona ganadora de cada grupo.

Avanza automaticamente; el TIEMPO en ms por entrada es editable (slider). Vistas: disparo digital
(umbral theta), solo ganadora, o activacion completa. Sin dependencias extra: http.server + numpy.

Uso:
    python hebbian/webapp.py                         # usa el model.npz mas reciente y el dataset de rectas
    python hebbian/webapp.py --model experiments/<run>/model.npz --dataset OTRO.npz --port 8000
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from competitive_net import CompetitiveLayer
from generate_lines import OUT as LINES_NPZ, SIZE

GRID = 50


def latest_model() -> Path | None:
    cands = sorted(Path("experiments").glob("**/model*.npz"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _u8_list(v: np.ndarray) -> list[int]:
    lo, hi = float(v.min()), float(v.max())
    n = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    return (n * 255).astype(np.uint8).tolist()


PAGE = """<!doctype html><html lang=es><head><meta charset=utf-8>
<title>Red competitiva - original vs negativo</title>
<style>
 body{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:0;padding:18px}
 h1{font-size:17px;margin:0 0 12px}
 .groups{display:flex;gap:30px;flex-wrap:wrap}
 .group{border:1px solid #333;border-radius:8px;padding:10px}
 .gtitle{font-size:14px;color:#cde;margin-bottom:8px;font-weight:600}
 .row{display:flex;gap:14px;align-items:flex-start}
 .panel{display:flex;flex-direction:column;align-items:center;gap:5px}
 .panel span{font-size:11px;color:#9aa}
 canvas{image-rendering:pixelated;background:#000;border:1px solid #333}
 .inp{width:150px;height:150px}.act{width:300px;height:300px}.rf{width:84px;height:84px}
 .controls{margin-top:16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:6px;padding:6px 12px;cursor:pointer}
 button:hover{background:#333}
 .readout{margin-top:12px;font-size:13px;color:#8fb}
 code{color:#fb8}
</style></head><body>
<h1>Red competitiva 784&rarr;2500 &mdash; ORIGINAL vs NEGATIVO fotografico (misma red)</h1>
<div class=groups>
  <div class=group>
    <div class=gtitle>Original</div>
    <div class=row>
      <div class=panel><canvas id=Oinp class=inp width=28 height=28></canvas><span>entrada</span></div>
      <div class=panel><canvas id=Oact class=act width=50 height=50></canvas><span>neuronas activas</span></div>
      <div class=panel><canvas id=Orf class=rf width=28 height=28></canvas><span>ganadora</span></div>
    </div>
  </div>
  <div class=group>
    <div class=gtitle>Negativo (foto)</div>
    <div class=row>
      <div class=panel><canvas id=Ninp class=inp width=28 height=28></canvas><span>entrada</span></div>
      <div class=panel><canvas id=Nact class=act width=50 height=50></canvas><span>neuronas activas</span></div>
      <div class=panel><canvas id=Nrf class=rf width=28 height=28></canvas><span>ganadora</span></div>
    </div>
  </div>
</div>
<div class=controls>
  <button id=play>Pausa</button>
  <button id=prev>&laquo; anterior</button>
  <button id=next>siguiente &raquo;</button>
  <label>ms por entrada: <b id=msval>500</b> <input id=ms type=range min=10 max=1000 step=10 value=500></label>
  <label>vista: <select id=mode>
     <option value=fire selected>disparo (digital)</option>
     <option value=winner>solo ganadora</option>
     <option value=full>activacion completa</option></select></label>
  <label>umbral &theta;: <b id=thrval>0.40</b> <input id=thr type=range min=0 max=0.49 step=0.01 value=0.40></label>
  <label><input id=shuffle type=checkbox> aleatorio</label>
</div>
<div class=readout id=readout></div>
<script>
const N_GRID=50, SIZE=28;
let total=0, idx=0, playing=true, timer=null, ms=500, order=null, mode='fire', thr=0.40;
const $=id=>document.getElementById(id);

function draw(canvas, arr, w, h, redAt){
  const ctx=canvas.getContext('2d'); const img=ctx.createImageData(w,h);
  for(let k=0;k<w*h;k++){const v=arr[k];img.data[4*k]=v;img.data[4*k+1]=v;img.data[4*k+2]=v;img.data[4*k+3]=255;}
  if(redAt!=null){const k=redAt;img.data[4*k]=255;img.data[4*k+1]=30;img.data[4*k+2]=30;img.data[4*k+3]=255;}
  ctx.putImageData(img,0,0);
}
function buildAct(araw){        // convierte activaciones crudas al mapa segun el modo
  const n=araw.length, arr=new Uint8Array(n); let fired=null;
  if(mode==='full'){ let lo=1e9,hi=-1e9; for(const v of araw){if(v<lo)lo=v;if(v>hi)hi=v;} const d=hi>lo?hi-lo:1;
    for(let k=0;k<n;k++) arr[k]=255*(araw[k]-lo)/d; }
  else if(mode==='fire'){ fired=0; for(let k=0;k<n;k++){ if(araw[k]>=thr){arr[k]=255;fired++;} } }
  return {arr, fired};                       // 'winner' -> todo negro salvo la ganadora (roja)
}
function drawGroup(p, g){
  draw($(p+'inp'), g.input, SIZE, SIZE, null);
  const {arr, fired}=buildAct(g.araw);
  draw($(p+'act'), arr, N_GRID, N_GRID, g.winner);
  draw($(p+'rf'), g.rf, SIZE, SIZE, null);
  return fired;
}
async function show(i){
  const d=await (await fetch('/api/frame?i='+i)).json();
  const fo=drawGroup('O', d.orig), fn=drawGroup('N', d.neg);
  const ex = mode==='fire' ? (' &nbsp; disparan: original <code>'+fo+'</code> vs negativo <code>'+fn+'</code>') : '';
  $('readout').innerHTML='entrada <code>'+(i+1)+'/'+total+'</code> &nbsp; ganadora original <code>#'+d.orig.winner
    +'</code> (act '+d.orig.wact.toFixed(3)+') &nbsp; ganadora negativo <code>#'+d.neg.winner
    +'</code> (act '+d.neg.wact.toFixed(3)+')'+ex;
}
function nextIndex(step){ if(order){let p=order.indexOf(idx);p=(p+step+total)%total;idx=order[p];} else idx=(idx+step+total)%total; }
function tick(){ nextIndex(1); show(idx); }
function restartTimer(){ if(timer)clearInterval(timer); if(playing)timer=setInterval(tick,ms); }
$('ms').oninput=e=>{ms=+e.target.value;$('msval').textContent=ms;restartTimer();};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pausa':'Reproducir';restartTimer();};
$('next').onclick=()=>{nextIndex(1);show(idx);};
$('prev').onclick=()=>{nextIndex(-1);show(idx);};
$('mode').onchange=e=>{mode=e.target.value;show(idx);};
$('thr').oninput=e=>{thr=+e.target.value;$('thrval').textContent=thr.toFixed(2);show(idx);};
$('shuffle').onchange=e=>{ if(e.target.checked){order=[...Array(total).keys()];for(let i=total-1;i>0;i--){const j=(Math.random()*(i+1))|0;[order[i],order[j]]=[order[j],order[i]];}} else order=null; };
(async()=>{const m=await (await fetch('/api/meta')).json();total=m.total;show(0);restartTimer();})();
</script></body></html>"""


def make_handler(layer: CompetitiveLayer, X: np.ndarray):
    def resp(x_raw: np.ndarray) -> dict:
        """Respuesta de la red a UNA entrada cruda (0..1): entrada, activaciones y ganadora."""
        xu = layer._normalize_vec(x_raw)
        a = layer.W @ xu
        w = int(a.argmax())
        return {
            "input": (x_raw * 255).astype(np.uint8).tolist(),
            "araw": np.round(a, 3).tolist(),         # activaciones crudas (umbral/normalizacion en cliente)
            "rf": _u8_list(layer.W[w]),
            "winner": w, "wr": w // GRID, "wc": w % GRID, "wact": float(a[w]),
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/api/meta":
                self._send(json.dumps({"total": len(X), "grid": GRID, "size": SIZE}).encode(), "application/json")
            elif u.path == "/api/frame":
                i = int(parse_qs(u.query).get("i", ["0"])[0]) % len(X)
                payload = {"index": i, "total": len(X),
                           "orig": resp(X[i]),
                           "neg": resp(1.0 - X[i])}      # negativo fotografico de la MISMA entrada
                self._send(json.dumps(payload).encode(), "application/json")
            else:
                self.send_error(404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="App web local para visualizar la red competitiva en accion")
    ap.add_argument("--model", type=Path, default=None, help="model.npz (por defecto, el mas reciente)")
    ap.add_argument("--dataset", type=Path, default=LINES_NPZ, help="npz con clave 'images'")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="no abrir el navegador automaticamente")
    args = ap.parse_args()

    model = args.model or latest_model()
    if model is None or not Path(model).exists():
        raise SystemExit("no encuentro ningun model.npz; entrena primero con hebbian/train.py")
    layer = CompetitiveLayer.load(model)

    imgs = np.load(args.dataset)["images"]
    X = imgs.reshape(len(imgs), -1).astype(np.float32)
    if X.max() > 1.0:
        X /= 255.0
    if layer.n_in != X.shape[1]:
        raise SystemExit(f"la red espera {layer.n_in} entradas y el dataset tiene {X.shape[1]}")

    handler = make_handler(layer, X)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"modelo:  {model}")
    print(f"dataset: {args.dataset} ({len(X)} entradas)  |  original vs negativo fotografico")
    print(f"sirviendo en {url}   (Ctrl+C para parar)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nparado.")
        server.shutdown()


if __name__ == "__main__":
    main()
