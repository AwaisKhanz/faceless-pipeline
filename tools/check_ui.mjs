/**
 * Render every view of lib/ui.html in a minimal DOM and report what came out.
 *
 * There is no browser in CI here, and shipping an interface that was only
 * syntax-checked is how you find out at the user's desk that a view throws on
 * first paint. This runs the real view functions against real API responses
 * from a live studio server, so anything that would blow up on load blows up
 * here instead.
 *
 *     node tools/check_ui.mjs                 # against a running studio
 *     node tools/check_ui.mjs --port 8765
 *
 * It is not a layout test — it cannot tell you something is ugly. It tells you
 * the code runs, the routes dispatch, and each view produces real elements.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const PORT = process.argv.includes('--port')
  ? process.argv[process.argv.indexOf('--port') + 1] : '8765';
const BASE = `http://127.0.0.1:${PORT}`;

// ─── the smallest DOM that ui.html actually uses ─────────────────────────────
class CL {
  constructor(n) { this.n = n; this.s = new Set(); }
  add(...c) { c.forEach(x => x && this.s.add(x)); this.sync(); }
  remove(...c) { c.forEach(x => this.s.delete(x)); this.sync(); }
  toggle(c, on) { on ? this.add(c) : this.remove(c); }
  contains(c) { return this.s.has(c); }
  sync() { this.n._class = [...this.s].join(' '); }
}
class Node {
  constructor(tag) {
    this.tag = tag; this.children = []; this.attrs = {}; this._class = '';
    this.style = { setProperty() { }, };
    this.dataset = {}; this.listeners = {}; this._text = '';
    this.classList = new CL(this);
    this.scrollHeight = 0; this.scrollTop = 0; this.clientHeight = 0;
  }
  get className() { return this._class; }
  set className(v) { this._class = v; this.classList.s = new Set(String(v).split(/\s+/).filter(Boolean)); }
  setAttribute(k, v) { this.attrs[k] = v; if (k === 'class') this.className = v; }
  getAttribute(k) { return this.attrs[k]; }
  addEventListener(k, fn) { (this.listeners[k] ||= []).push(fn); }
  append(...kids) { for (const k of kids) this.children.push(k); }
  get textContent() {
    return this._text + this.children.map(c => c.textContent ?? String(c)).join('');
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  set innerHTML(v) { this.children = []; this._text = ''; }
  get innerText() { return this.textContent; }
  querySelectorAll() { return []; }
  remove() { }
  play() { return Promise.resolve(); }
  pause() { }
  focus() { }
  get all() {
    return [this, ...this.children.filter(c => c instanceof Node).flatMap(c => c.all)];
  }
}
class TextNode { constructor(t) { this.textContent = String(t); this.nodeType = 3; } }

const REG = {};
const doc = {
  createElement: t => { const n = new Node(t); n.nodeType = 1; return n; },
  createTextNode: t => new TextNode(t),
  documentElement: Object.assign(new Node('html'), { dataset: {} }),
  querySelector: sel => REG[sel] ||= new Node('div'),
  querySelectorAll: () => [],
  addEventListener() { },
};

// ─── globals ui.html expects ─────────────────────────────────────────────────
const calls = [];
globalThis.document = doc;
globalThis.window = { scrollTo() { }, open() { } };
globalThis.location = { hash: '#/dashboard' };
globalThis.localStorage = { getItem: () => null, setItem() { } };
// node 22 defines navigator as a getter-only global; redefine it.
Object.defineProperty(globalThis, 'navigator',
  { value: { clipboard: { writeText() { } } }, configurable: true });
globalThis.Audio = class { play() { } };
globalThis.addEventListener = () => { };
// Same trap as fetch: grab the real one first. The poll loop reschedules
// itself every 900ms, so long timers are swallowed to keep the test finite.
const realTimeout = globalThis.setTimeout.bind(globalThis);
globalThis.setTimeout = (fn, ms) => (ms > 500 ? 0 : realTimeout(fn, ms));
globalThis.clearTimeout = () => { };
globalThis.confirm = () => true;
// Capture the REAL fetch before shadowing it — assigning globalThis.__fetch
// afterwards would just point it at the replacement and recurse forever.
const realFetch = globalThis.fetch.bind(globalThis);
globalThis.__fetch = realFetch;
globalThis.fetch = async (ep, opt) => {
  calls.push(ep);
  const r = await realFetch(BASE + ep, opt);
  return { json: () => r.json() };
};

// ─── load the script out of the page ─────────────────────────────────────────
const html = readFileSync(path.join(ROOT, 'lib', 'ui.html'), 'utf8');
const js = html.match(/<script>([\s\S]*)<\/script>/)[1];

// Expose the view functions so we can call them one at a time.
const mod = js + `
;globalThis.__views = { viewDashboard, viewProject, viewRun, viewNew,
                        viewVoices, viewSettings, viewReview, route,
                        fractions, overall, fmtT, ago };`;

let fail = 0;
try {
  // Top level runs route() and poll(); both are async and harmless here.
  await (new Function(mod))();
} catch (e) {
  console.log('  !! the script threw while loading:', e.message);
  process.exit(1);
}
const V = globalThis.__views;

// ─── pure logic ──────────────────────────────────────────────────────────────
console.log('\n  pure functions');
const st = { scenes: 100, assets: 100, languages: { en: { sheets: true, voice_n: 50, render: false } } };
const f = V.fractions(st, st.languages.en);
const okF = f.sheets === 1 && f.visuals === 1 && f.voice === 0.5 && f.render === 0;
console.log(`  ${okF ? 'ok' : '!!'}  fractions  ${JSON.stringify(f)}`); fail += !okF;

const cases = [
  [{ languages: { a: { render: true }, b: { render: true } } }, 'complete'],
  [{ languages: { a: { render: true }, b: { render: false } } }, '1/2 rendered'],
  [{ assets: 5, languages: { a: { render: false, voice_n: 0 } } }, 'in progress'],
  [{ assets: 0, languages: { a: { render: false, voice_n: 0 } } }, 'not started'],
];
for (const [inp, want] of cases) {
  const got = V.overall(inp).text;
  console.log(`  ${got === want ? 'ok' : '!!'}  overall -> ${got}`); fail += got !== want;
}
const tt = [[45, '45s'], [90, '1m 30s'], [3700, '1h 01m']];
for (const [s, want] of tt) {
  const got = V.fmtT(s);
  console.log(`  ${got === want ? 'ok' : '!!'}  fmtT(${s}) -> ${got}`); fail += got !== want;
}

// ─── each view, against the live server ──────────────────────────────────────
console.log('\n  views (rendered against the running server)');
const view = new Node('div');
REG['#view'] = view;

const projects = await (await globalThis.__fetch(BASE + '/api/projects')).json();
const firstId = projects.projects?.[0]?.id;

const run = async (name, fn, arg) => {
  REG['#view'] = new Node('div');
  try {
    await fn(arg);
    const nodes = REG['#view'].all.length;
    const text = REG['#view'].textContent.replace(/\s+/g, ' ').trim();
    const ok = nodes > 3;
    console.log(`  ${ok ? 'ok' : '!!'}  ${name.padEnd(14)} ${String(nodes).padStart(4)} nodes  ${text.slice(0, 62)}`);
    fail += !ok;
  } catch (e) {
    console.log(`  !!  ${name.padEnd(14)} threw: ${e.message}`);
    fail++;
  }
};

await run('dashboard', V.viewDashboard);
if (firstId) await run('project', V.viewProject, firstId);
if (firstId) await run('review', V.viewReview, firstId);
await run('run/activity', V.viewRun);
await run('new', V.viewNew);
await run('voices', V.viewVoices);
await run('settings', V.viewSettings);

console.log(`\n  ${fail === 0 ? 'ALL PASS' : fail + ' FAILURE(S)'} — ${calls.length} API calls made`);
process.exit(fail ? 1 : 0);
