"""Build a self-contained approval sheet: every stock pick, side by side with
its narration line. Click the bad ones, copy the command it prints, re-run."""
from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """
*{box-sizing:border-box} body{margin:0;background:#14161a;color:#e8eaed;
font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
header{position:sticky;top:0;z-index:10;background:#1c1f25;border-bottom:1px solid #2c313a;
padding:16px 24px;display:flex;gap:20px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:600}
.meta{color:#9aa0a6;font-size:13px}
button{background:#2f6fed;color:#fff;border:0;border-radius:6px;padding:9px 16px;
font-size:14px;font-weight:600;cursor:pointer}
button.ghost{background:#2c313a}
#cmd{width:100%;margin-top:12px;background:#0e1013;border:1px solid #2c313a;color:#7ee787;
border-radius:6px;padding:12px;font:13px ui-monospace,Menlo,monospace;display:none}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));
gap:16px;padding:24px}
.card{background:#1c1f25;border:2px solid #2c313a;border-radius:10px;overflow:hidden;
cursor:pointer;transition:border-color .12s}
.card:hover{border-color:#4a5160}
.card.redo{border-color:#f85149;background:#2a1a1c}
.card.redo .thumb::after{content:'REDO';position:absolute;top:10px;right:10px;
background:#f85149;color:#fff;font-size:11px;font-weight:700;padding:4px 8px;border-radius:4px}
.card.hero{border-color:#d29922}
.card.hero.redo{border-color:#f85149}
.thumb{position:relative;aspect-ratio:16/9;background:#000;overflow:hidden}
.thumb img,.thumb video{width:100%;height:100%;object-fit:cover;display:block}
.tag{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.78);font-size:11px;
font-weight:700;padding:4px 8px;border-radius:4px;letter-spacing:.4px}
.tag.v{background:#8957e5}
.hero .tag.h{position:absolute;bottom:10px;left:10px;background:#d29922;color:#14161a}
.body{padding:12px 14px}
.narr{font-size:13.5px;color:#e8eaed;margin:0 0 8px}
.q{font:12px ui-monospace,Menlo,monospace;color:#7d8590;word-break:break-word}
.missing{display:flex;align-items:center;justify-content:center;height:100%;
color:#f85149;font-size:13px;text-align:center;padding:16px}
"""

JS = """
const st = new Set();
document.querySelectorAll('.card').forEach(c=>{
  c.onclick = e => {
    if (e.target.tagName === 'VIDEO') return;
    const n = +c.dataset.n;
    c.classList.toggle('redo');
    c.classList.contains('redo') ? st.add(n) : st.delete(n);
    upd();
  };
});
function upd(){
  const box = document.getElementById('cmd');
  const cnt = document.getElementById('cnt');
  cnt.textContent = st.size ? st.size + ' marked for redo' : 'none marked';
  if (!st.size){ box.style.display='none'; return; }
  const list=[...st].sort((a,b)=>a-b).join(',');
  box.style.display='block';
  box.value = 'python3 make_video.py stock --sheet ' + SHEET +
              ' --lang ' + LANG + ' --redo ' + list;
}
function copyCmd(){
  const b=document.getElementById('cmd');
  if(!b.value) return;
  b.select(); document.execCommand('copy');
  const btn=document.getElementById('cp'); const t=btn.textContent;
  btn.textContent='Copied'; setTimeout(()=>btn.textContent=t,1200);
}
function markAll(){
  document.querySelectorAll('.card').forEach(c=>{c.classList.add('redo');st.add(+c.dataset.n);});
  upd();
}
function clearAll(){
  document.querySelectorAll('.card').forEach(c=>c.classList.remove('redo'));
  st.clear(); upd();
}
"""


def build(scenes, assets: dict[int, dict], out: Path, sheet: str, lang: str,
          picks: dict[int, int] | None = None) -> Path:
    picks = picks or {}
    cards = []
    for s in scenes:
        a = assets.get(s.n)
        classes = "card" + (" hero" if s.hero else "")
        if a:
            rel = Path(a["path"]).resolve().as_uri()
            inner = (f'<video src="{rel}" muted preload="metadata"></video>'
                     if a["path"].lower().endswith(".mp4")
                     else f'<img src="{rel}" loading="lazy">')
            take = picks.get(s.n, 0)
            take_lbl = f" · take {take + 1}" if take else ""
            sub = f'{html.escape(a["src"])}{take_lbl}'
        else:
            inner = '<div class="missing">no match found<br>edit the ALT query</div>'
            sub = "missing"

        cards.append(f"""
<div class="{classes}" data-n="{s.n}">
  <div class="thumb">{inner}
    <span class="tag {'v' if s.media == 'VIDEO' else ''}">S{s.n} · {s.media} · {sub}</span>
    {'<span class="tag h">HERO</span>' if s.hero else ''}
  </div>
  <div class="body">
    <p class="narr">{html.escape(s.narration[:190])}{'…' if len(s.narration) > 190 else ''}</p>
    <div class="q">{html.escape(s.query)}</div>
  </div>
</div>""")

    n_hero = sum(1 for s in scenes if s.hero)
    n_miss = sum(1 for s in scenes if s.n not in assets)
    doc = f"""<!doctype html><meta charset="utf-8">
<title>Approval sheet · {html.escape(sheet)} · {lang.upper()}</title>
<style>{CSS}</style>
<header>
  <h1>Approval sheet — {html.escape(Path(sheet).name)} · {lang.upper()}</h1>
  <span class="meta">{len(scenes)} scenes · {n_hero} hero · {n_miss} missing ·
    <b id="cnt">none marked</b></span>
  <span style="flex:1"></span>
  <button class="ghost" onclick="markAll()">Mark all</button>
  <button class="ghost" onclick="clearAll()">Clear</button>
  <button id="cp" onclick="copyCmd()">Copy redo command</button>
  <textarea id="cmd" rows="2" readonly></textarea>
</header>
<p style="padding:20px 24px 0;color:#9aa0a6;margin:0">
  Click any card that does not fit the line. Then copy the command at the top and run it —
  it pulls the next-best match for just those scenes and rebuilds this page.
  Gold border = hero scene (recurring character or title card), worth checking closely.
</p>
<div class="grid">{''.join(cards)}</div>
<script>const SHEET={json.dumps(sheet)}, LANG={json.dumps(lang)};{JS}</script>
"""
    out.write_text(doc, encoding="utf-8")
    return out
