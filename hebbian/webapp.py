"""App web LOCAL para visualizar la red competitiva en accion.

Muestra a la vez, para cada entrada del dataset:
  - la imagen de ENTRADA (28x28),
  - el MAPA 50x50 de ACTIVACION de las neuronas (las que se activaron brillan; la ganadora en rojo),
  - el campo receptivo de la neurona ganadora (para ver "que patron" gano).

Avanza automaticamente de una entrada a la siguiente; el TIEMPO en ms por entrada es un parametro
editable desde la propia pagina (slider). Sin dependencias extra: solo http.server + numpy.

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
    cands = sorted(Path("experiments").glob("hebbian_lines_*/model.npz"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _u8_list(v: np.ndarray) -> list[int]:
    lo, hi = float(v.min()), float(v.max())
    n = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    return (n * 255).astype(np.uint8).tolist()


PAGE = """<!doctype html><html lang=es><head><meta charset=utf-8>
<title>Red competitiva - visualizacion</title>
<style>
 body{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:0;padding:20px}
 h1{font-size:18px;font-weight:600;margin:0 0 14px}
 .row{display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start}
 .panel{display:flex;flex-direction:column;align-items:center;gap:6px}
 .panel span{font-size:12px;color:#9aa}
 canvas{image-rendering:pixelated;background:#000;border:1px solid #333}
 #inp{width:224px;height:224px}
 #act{width:400px;height:400px}
 #rf{width:140px;height:140px}
 .controls{margin-top:18px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
 button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:6px;padding:6px 12px;cursor:pointer}
 button:hover{background:#333}
 input[type=range]{vertical-align:middle}
 .readout{font-size:13px;color:#8fb;min-width:340px}
 code{color:#fb8}
</style></head><body>
<h1>Red competitiva 784 &rarr; 2500 (50&times;50) &mdash; entrada vs. neuronas activadas</h1>
<div class=row>
  <div class=panel><canvas id=inp width=28 height=28></canvas><span>entrada 28&times;28</span></div>
  <div class=panel><canvas id=act width=50 height=50></canvas><span id=actcap>disparo digital: blancas = disparan, ganadora en rojo</span></div>
  <div class=panel><canvas id=rf width=28 height=28></canvas><span>campo receptivo ganadora</span></div>
</div>
<div class=controls>
  <button id=play>Pausa</button>
  <button id=prev>&laquo; anterior</button>
  <button id=next>siguiente &raquo;</button>
  <label>ms por entrada: <b id=msval>500</b>
    <input id=ms type=range min=10 max=1000 step=10 value=500></label>
  <label>vista:
    <select id=mode>
      <option value=fire selected>disparo (digital)</option>
      <option value=winner>solo ganadora</option>
      <option value=full>activacion completa</option>
    </select></label>
  <label>umbral &theta;: <b id=thrval>0.40</b>
    <input id=thr type=range min=0 max=0.49 step=0.01 value=0.40></label>
  <label><input id=shuffle type=checkbox> aleatorio</label>
</div>
<div class=readout id=readout></div>
<script>
const N_GRID=50, SIZE=28;
let total=0, idx=0, playing=true, timer=null, ms=500, order=null, mode='fire', thr=0.40;
const $=id=>document.getElementById(id);

function draw(canvas, arr, w, h, redAt){
  const ctx=canvas.getContext('2d');
  const img=ctx.createImageData(w,h);
  for(let k=0;k<w*h;k++){const v=arr[k];img.data[4*k]=v;img.data[4*k+1]=v;img.data[4*k+2]=v;img.data[4*k+3]=255;}
  if(redAt!=null){const k=redAt;img.data[4*k]=255;img.data[4*k+1]=30;img.data[4*k+2]=30;img.data[4*k+3]=255;}
  ctx.putImageData(img,0,0);
}

async function show(i){
  const r=await fetch('/api/frame?i='+i); const d=await r.json();
  draw($('inp'), d.input, SIZE, SIZE, null);
  let actArr, fired=null;
  if(mode==='full'){ actArr=d.act; }
  else if(mode==='winner'){ actArr=new Uint8Array(N_GRID*N_GRID); }
  else { // disparo digital: blanco si activacion >= umbral
    actArr=new Uint8Array(N_GRID*N_GRID); fired=0;
    for(let k=0;k<actArr.length;k++){ if(d.araw[k]>=thr){actArr[k]=255;fired++;} }
  }
  draw($('act'), actArr, N_GRID, N_GRID, d.winner);
  draw($('rf'), d.rf, SIZE, SIZE, null);
  let extra = (fired!=null) ? ' &nbsp; neuronas que disparan (&theta;='+thr.toFixed(2)+'): <code>'+fired+'</code>' : '';
  $('readout').innerHTML='entrada <code>'+ (i+1) +'/'+total+'</code> &nbsp; ganadora: neurona <code>#'
    +d.winner+'</code> (fila '+d.wr+', col '+d.wc+') &nbsp; activacion max <code>'+d.wact.toFixed(3)+'</code>'+extra;
}

function nextIndex(step){
  if(order){ // recorre en orden aleatorio precalculado
    let p=order.indexOf(idx); p=(p+step+total)%total; idx=order[p];
  } else { idx=(idx+step+total)%total; }
}
function tick(){ nextIndex(1); show(idx); }
function restartTimer(){ if(timer)clearInterval(timer); if(playing)timer=setInterval(tick,ms); }

$('ms').oninput=e=>{ms=+e.target.value;$('msval').textContent=ms;restartTimer();};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pausa':'Reproducir';restartTimer();};
const CAPS={fire:'disparo digital: blancas = disparan, ganadora en rojo',winner:'solo la ganadora (rojo), resto en negro',full:'activacion completa (gradiente), ganadora en rojo'};
$('mode').onchange=e=>{mode=e.target.value;$('actcap').innerHTML=CAPS[mode];show(idx);};
$('thr').oninput=e=>{thr=+e.target.value;$('thrval').textContent=thr.toFixed(2);if(mode==='fire')show(idx);};
$('next').onclick=()=>{nextIndex(1);show(idx);};
$('prev').onclick=()=>{nextIndex(-1);show(idx);};
$('shuffle').onchange=e=>{
  if(e.target.checked){order=[...Array(total).keys()];for(let i=total-1;i>0;i--){const j=(Math.random()*(i+1))|0;[order[i],order[j]]=[order[j],order[i]];}}
  else order=null;
};

(async()=>{const m=await (await fetch('/api/meta')).json();total=m.total;show(0);restartTimer();})();
</script></body></html>"""


def make_handler(layer: CompetitiveLayer, X: np.ndarray, Xn: np.ndarray):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silencia el log por request
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
                a = layer.W @ Xn[i]                      # activacion de las 2500 neuronas
                winner = int(a.argmax())
                payload = {
                    "index": i,
                    "input": (X[i] * 255).astype(np.uint8).tolist(),
                    "act": _u8_list(a),
                    "rf": _u8_list(layer.W[winner]),
                    "winner": winner,
                    "wr": winner // GRID,
                    "wc": winner % GRID,
                    "wact": float(a[winner]),
                    "araw": np.round(a, 3).tolist(),         # activaciones crudas (para umbral digital en cliente)
                }
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
    Xn = layer._normalize_rows(X)

    handler = make_handler(layer, X, Xn)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"modelo:  {model}")
    print(f"dataset: {args.dataset} ({len(X)} entradas)")
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
