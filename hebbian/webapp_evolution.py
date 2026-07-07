"""Visor de EVOLUCION: para UNA imagen fija, reproduce como cambia el conjunto de neuronas que
disparan a lo largo del entrenamiento (fotogramas grabados por train_one.py).

Mapa 50x50 con RASTRO DE PERSISTENCIA (integrador con fuga por neurona):
  - mientras DISPARA, su color se aclara hacia BLANCO (empieza gris oscuro).
  - mientras esta APAGADA, su color se oscurece hacia NEGRO.
  - las que parpadean se quedan en gris intermedio.
  Asi resaltan las mas persistentes: blanco = lleva mucho disparando, negro = lleva mucho apagada.
  - azul = neurona ganadora (argmax) de este paso.

Parametros editables: 'ms por paso', el umbral theta, y 'frames -> blanco/negro' (velocidad del
rastro). Sin dependencias extra.

Uso:
    python hebbian/webapp_evolution.py --frames experiments/one_evo/frames.npz --port 8000
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

GRID = 50
SIZE = 28

PAGE = """<!doctype html><html lang=es><head><meta charset=utf-8>
<title>Evolucion del disparo (1 imagen)</title>
<style>
 body{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:0;padding:18px}
 h1{font-size:17px;margin:0 0 12px}
 .row{display:flex;gap:22px;align-items:flex-start;flex-wrap:wrap}
 .panel{display:flex;flex-direction:column;align-items:center;gap:5px}
 .panel span{font-size:12px;color:#9aa}
 canvas{image-rendering:pixelated;background:#000;border:1px solid #333}
 #inp{width:170px;height:170px}#map{width:420px;height:420px}
 .controls{margin-top:16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:6px;padding:6px 12px;cursor:pointer}
 button:hover{background:#333}
 .readout{margin-top:12px;font-size:13px}
 .leg{font-size:12px;color:#9aa} .leg b{padding:1px 5px;border-radius:3px}
 code{color:#fb8}
</style></head><body>
<h1>Evolucion del disparo ante UNA imagen &mdash; rastro de persistencia (mas tiempo disparando = mas claro)</h1>
<div class=row>
  <div class=panel><canvas id=inp width=28 height=28></canvas><span>imagen (fija)</span></div>
  <div class=panel><canvas id=map width=50 height=50></canvas><span>neuronas que disparan (50&times;50)</span></div>
</div>
<div class=leg>
  <b style="background:#111;color:#888">negro</b> lleva apagada &rarr;
  <b style="background:#7a7a7a">gris</b> parpadea &rarr;
  <b style="background:#e8e8e8;color:#000">blanco</b> lleva disparando &nbsp;&nbsp;
  <b style="background:#3060ff">azul</b> ganadora
</div>
<div class=controls>
  <button id=play>Pausa</button>
  <button id=prev>&laquo; paso ant.</button>
  <button id=next>paso sig. &raquo;</button>
  <label>ms por paso: <b id=msval>350</b> <input id=ms type=range min=20 max=1500 step=10 value=350></label>
  <label>frames &rarr; blanco/negro: <b id=persval>40</b> <input id=pers type=range min=2 max=150 step=1 value=40></label>
  <label>umbral &theta;: <b id=thrval>0.40</b> <input id=thr type=range min=0 max=0.49 step=0.01 value=0.40></label>
  <label><input id=loop type=checkbox checked> repetir</label>
</div>
<div class=readout id=readout></div>
<script>
const N=50, SIZE=28;
let total=0, idx=0, playing=true, timer=null, ms=350, thr=0.40, pers=40;
const $=id=>document.getElementById(id);

function drawGray(canvas, arr){
  const ctx=canvas.getContext('2d'), img=ctx.createImageData(SIZE,SIZE);
  for(let k=0;k<SIZE*SIZE;k++){const v=arr[k];img.data[4*k]=v;img.data[4*k+1]=v;img.data[4*k+2]=v;img.data[4*k+3]=255;}
  ctx.putImageData(img,0,0);
}
function drawMap(bright, winner){
  const ctx=$('map').getContext('2d'), img=ctx.createImageData(N,N);
  let persist=0;
  for(let k=0;k<N*N;k++){
    const v=bright[k]; img.data[4*k]=v;img.data[4*k+1]=v;img.data[4*k+2]=v;img.data[4*k+3]=255;
    if(v>=230) persist++;
  }
  const w=4*winner; img.data[w]=48;img.data[w+1]=96;img.data[w+2]=255;img.data[w+3]=255;
  ctx.putImageData(img,0,0);
  return {persist};
}
async function show(i){
  const d=await (await fetch('/api/frame?i='+i+'&thr='+thr+'&pers='+pers)).json();
  drawGray($('inp'), d.input);
  const s=drawMap(d.bright, d.winner);
  $('readout').innerHTML='paso <code>'+(i+1)+'/'+total+'</code> &nbsp; epoca <code>'+d.epoch
    +'</code> &nbsp; disparan ahora <code>'+d.fired+'</code> &nbsp; '
    +'<span style=color:#eee>persistentes (blancas) <code>'+s.persist+'</code></span>';
}
function step(k){ idx+=k; if(idx>=total){ if($('loop').checked) idx=0; else {idx=total-1; playing=false; $('play').textContent='Reproducir'; }} if(idx<0) idx=total-1; show(idx); }
function restart(){ if(timer)clearInterval(timer); if(playing)timer=setInterval(()=>step(1),ms); }
$('ms').oninput=e=>{ms=+e.target.value;$('msval').textContent=ms;restart();};
$('pers').oninput=e=>{pers=+e.target.value;$('persval').textContent=pers;show(idx);};
$('thr').oninput=e=>{thr=+e.target.value;$('thrval').textContent=thr.toFixed(2);show(idx);};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pausa':'Reproducir';restart();};
$('next').onclick=()=>{playing=false;$('play').textContent='Reproducir';step(1);};
$('prev').onclick=()=>{playing=false;$('play').textContent='Reproducir';step(-1);};
(async()=>{const m=await (await fetch('/api/meta')).json();total=m.total;thr=m.theta;
  $('thr').value=thr;$('thrval').textContent=thr.toFixed(2);show(0);restart();})();
</script></body></html>"""


def make_handler(inp, acts, epochs, theta):
    acts_f = acts.astype(np.float32)
    n = acts_f.shape[1]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/api/meta":
                self._send(json.dumps({"total": len(acts_f), "theta": float(theta)}).encode(), "application/json")
            elif u.path == "/api/frame":
                i = int(q.get("i", ["0"])[0]) % len(acts_f)
                thr = float(q.get("thr", [str(theta)])[0])
                pers = max(int(q.get("pers", ["40"])[0]), 1)
                F = acts_f[:i + 1] >= thr                       # disparo por frame hasta i
                inc = 1.0 / pers
                # integrador con fuga: sube si dispara, baja si no; asi el rastro marca la persistencia
                val = np.zeros(n, dtype=np.float32)
                for j in range(i + 1):
                    val += np.where(F[j], inc, -inc)
                    np.clip(val, 0.0, 1.0, out=val)
                bright = (val * 255).astype(int)
                self._send(json.dumps({
                    "input": inp.tolist(),
                    "bright": bright.tolist(),
                    "fired": int(F[i].sum()),
                    "winner": int(acts_f[i].argmax()),
                    "epoch": int(epochs[i]),
                }).encode(), "application/json")
            else:
                self.send_error(404)
    return H


def main() -> None:
    ap = argparse.ArgumentParser(description="Visor de evolucion del disparo (1 imagen)")
    ap.add_argument("--frames", type=Path, default=Path("experiments/one_evo/frames.npz"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    d = np.load(args.frames)
    acts, epochs = d["acts"], d["epochs"]
    inp = d["input"].astype(int)
    theta = float(d["theta"])
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(inp, acts, epochs, theta))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(acts)} fotogramas (imagen #{int(d['img_index'])})  theta={theta}  sirviendo en {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
