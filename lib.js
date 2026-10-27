/* Component library: formatting, charts, tables, tabs, cards. Everything draws with d3 + DOM; data values are always inserted as text. */
window.L = (() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const el = (t, c, x) => { const e = document.createElement(t); if (c) e.className = c; if (x != null) e.textContent = x; return e; };
  const F = { int: d3.format(','), f1: d3.format('.1f'), f2: d3.format('.2f'), f3: d3.format('.3f'), p0: d3.format('.0%'), p1: d3.format('.1%'), pc: (v) => d3.format('.1f')(v) + '%',
    sgn: (v) => (v > 0 ? '+' : v < 0 ? '−' : '') + d3.format('.1f')(Math.abs(v)), compact: (v) => new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 }).format(v) };
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const SER = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)'];

  // tooltip
  const tipEl = () => $('#tip');
  const showTip = (ev, lines) => { const t = tipEl(); t.replaceChildren(...lines.map((l) => el('div', null, l))); t.style.display = 'block';
    const w = t.offsetWidth, h = t.offsetHeight; t.style.left = Math.max(6, Math.min(ev.clientX + 14, innerWidth - w - 6)) + 'px'; t.style.top = Math.max(6, Math.min(ev.clientY + 16, innerHeight - h - 6)) + 'px'; };
  const hideTip = () => { tipEl().style.display = 'none'; };

  const width = (host, min = 280) => Math.max(min, host.clientWidth || host.parentElement?.clientWidth || min);
  const svgIn = (host, W, H, label) => d3.select(host).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('role', 'img').attr('aria-label', label);

  // horizontal bars (HTML): o = {label, value, max, fmt, color(r), best}
  function hbars(host, data, o) { host.replaceChildren(); const mx = o.max ?? d3.max(data, o.value);
    data.forEach((r) => { const row = el('div', 'hb' + (o.best && o.best(r) ? ' best' : '')); row.append(el('span', 'l', o.label(r)));
      const t = el('span', 't'), b = el('span', 'b'); b.style.width = Math.max(0, 100 * o.value(r) / mx) + '%'; b.style.background = o.color ? o.color(r) : 'var(--accent)'; t.append(b);
      row.append(t, el('span', 'v', o.fmt(o.value(r)))); if (o.tip) { row.addEventListener('mousemove', (e) => showTip(e, o.tip(r))); row.addEventListener('mouseleave', hideTip); } host.append(row); }); }

  // columns: o = {x, y, yfmt, hi, tip, h, lw, ymax, ymin, label, color}
  function cols(host, data, o) { host.replaceChildren(); const W = width(host), H = o.h || 230, m = { t: 14, r: 8, b: 38, l: 46 }, svg = svgIn(host, W, H, o.label);
    const x = d3.scaleBand().domain(data.map((d, i) => i)).range([m.l, W - m.r]).padding(0.26), y0 = o.ymin ?? 0, y = d3.scaleLinear().domain([y0, o.ymax ?? d3.max(data, o.y) * 1.12]).range([H - m.b, m.t]);
    y.ticks(4).forEach((t) => { svg.append('line').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y(t)).attr('y2', y(t)).style('stroke', 'var(--grid)'); svg.append('text').attr('x', m.l - 7).attr('y', y(t) + 4).attr('text-anchor', 'end').text(o.yfmt(t)); });
    const step = Math.ceil(data.length * (o.lw || 30) / (W - m.l - m.r));
    data.forEach((d, i) => { svg.append('rect').attr('x', x(i)).attr('width', x.bandwidth()).attr('y', y(Math.max(y0, o.y(d)))).attr('height', Math.max(0, y(y0) - y(Math.max(y0, o.y(d))))).attr('rx', 4)
        .style('fill', o.color ? o.color(d) : (o.hi && o.hi(d) ? 'var(--accent)' : 'var(--mutedbar)')).on('mousemove', (ev) => showTip(ev, [o.x(d), o.tip(d)])).on('mouseleave', hideTip);
      if (i % step === 0) svg.append('text').attr('x', x(i) + x.bandwidth() / 2).attr('y', H - m.b + 15).attr('text-anchor', 'middle').text(o.x(d)); }); }

  function line(host, data, o) { host.replaceChildren(); const W = width(host), H = o.h || 230, m = { t: 14, r: 14, b: 30, l: 46 }, svg = svgIn(host, W, H, o.label);
    const x = d3.scaleLinear(d3.extent(data, o.x), [m.l, W - m.r]), y = d3.scaleLinear(o.yd, [H - m.b, m.t]);
    y.ticks(4).forEach((t) => { svg.append('line').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y(t)).attr('y2', y(t)).style('stroke', 'var(--grid)'); svg.append('text').attr('x', m.l - 7).attr('y', y(t) + 4).attr('text-anchor', 'end').text(o.yfmt(t)); });
    o.xticks.forEach((t) => svg.append('text').attr('x', x(t)).attr('y', H - 9).attr('text-anchor', 'middle').text(o.xfmt(t)));
    svg.append('path').datum(data).attr('d', d3.line().x((r) => x(o.x(r))).y((r) => y(o.y(r)))).style('fill', 'none').style('stroke', 'var(--accent)').attr('stroke-width', 2.4).attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round');
    data.forEach((r) => svg.append('circle').attr('cx', x(o.x(r))).attr('cy', y(o.y(r))).attr('r', 6).style('fill', 'var(--accent)').attr('fill-opacity', 0.001).on('mousemove', (ev) => showTip(ev, o.tip(r))).on('mouseleave', hideTip)); }

  // segmented control
  function seg(options, value, onChange, label) { const g = el('div', 'seg'); g.setAttribute('role', 'group'); if (label) g.setAttribute('aria-label', label);
    const paint = () => $$('button', g).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === g.dataset.value)));
    options.forEach(([v, t]) => { const b = el('button', null, t); b.dataset.v = v; b.type = 'button'; b.addEventListener('click', () => { g.dataset.value = v; paint(); onChange(v); }); g.append(b); }); g.dataset.value = value; paint(); return g; }

  // card
  function card(title, sub, parent, cls) { const c = el('div', 'card' + (cls ? ' ' + cls : '')), h = el('div', 'card-h'); if (title) h.append(el('h3', null, title)); if (sub) h.append(el('span', 'muted', sub)); const grow = el('span', 'grow'); h.append(grow);
    const body = el('div'); body.style.cssText = 'display:flex;flex-direction:column;gap:12px;min-width:0'; c.append(h, body); if (parent) parent.append(c); return { root: c, head: h, actions: grow, body }; }

  // sortable table: spec = {cols:[{k,label,l,fmt,v,cls,sortv}], rows, sort:[key,dir], rowCls, csv}
  function table(host, spec) { host.replaceChildren(); const wrap = el('div', 'tw'), t = el('table'), thead = el('thead'), hr = el('tr'), tb = el('tbody'); if (spec.maxH) wrap.style.maxHeight = spec.maxH; let sk = spec.sort?.[0], dir = spec.sort?.[1] || 'descending';
    const val = (c, r) => (c.v ? c.v(r) : r[c.k]);
    const key = (c, r) => (c.sortv ? c.sortv(r) : val(c, r));
    const paint = () => { const col = spec.cols.find((c) => c.k === sk); let rows = [...spec.rows];
      if (col) { const has = rows.filter((r) => key(col, r) != null), none = rows.filter((r) => key(col, r) == null);
        has.sort((a, b) => { const x = key(col, a), y = key(col, b); return typeof x === 'string' ? x.localeCompare(y) : x - y; }); if (dir === 'descending') has.reverse(); rows = [...has, ...none]; }
      tb.replaceChildren(); rows.forEach((r) => { const tr = el('tr', spec.rowCls ? spec.rowCls(r, rows) : ''); spec.cols.forEach((c) => { const v = val(c, r), td = el('td', c.l ? 'l' : '');
          if (c.node) { const n = c.node(r); if (n) td.append(n); } else td.textContent = v == null ? '' : (c.fmt ? c.fmt(v, r) : String(v)); const k = c.cls && c.cls(r); if (k) td.className += ' ' + k; tr.append(td); }); tb.append(tr); });
      $$('th', hr).forEach((th, i) => th.setAttribute('aria-sort', spec.cols[i].k === sk ? dir : 'none')); host._rows = rows; };
    spec.cols.forEach((c) => { const th = el('th', c.l ? 'l' : '', c.label); th.addEventListener('click', () => { if (sk === c.k) dir = dir === 'descending' ? 'ascending' : 'descending'; else { sk = c.k; dir = c.l ? 'ascending' : 'descending'; } paint(); }); hr.append(th); });
    thead.append(hr); t.append(thead, tb); wrap.append(t); host.append(wrap); paint(); host._spec = spec; return host; }
  function csvButton(getTable, name) { const b = el('button', 'btn small', 'Download CSV'); b.type = 'button'; b.addEventListener('click', () => { const h = getTable(); if (!h || !h._spec) return; const q = (s) => '"' + String(s ?? '').replace(/"/g, '""') + '"';
      const head = h._spec.cols.map((c) => q(c.label)).join(','), body = h._rows.map((r) => h._spec.cols.map((c) => { const v = c.v ? c.v(r) : r[c.k]; return q(v == null ? '' : v); }).join(','));
      const a = el('a'); a.href = URL.createObjectURL(new Blob([head + '\n' + body.join('\n')], { type: 'text/csv' })); a.download = name + '.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }); return b; }

  // tabs: items = [{id,label,build(panel)->draw()}]
  function tabs(items, active, onChange) { const bar = el('div', 'tabs'), wrap = el('div'); bar.setAttribute('role', 'tablist'); const state = {}, panels = {};
    items.forEach((it) => { const b = el('button', null, it.label); b.type = 'button'; b.id = 'tab-' + it.id; b.setAttribute('role', 'tab'); b.addEventListener('click', () => show(it.id, true)); bar.append(b);
      const p = el('div', 'panel'); p.id = 'panel-' + it.id; p.setAttribute('role', 'tabpanel'); p.hidden = true; panels[it.id] = p; wrap.append(p); });
    bar.addEventListener('keydown', (e) => { if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return; const i = items.findIndex((x) => x.id === state.cur); show(items[(i + (e.key === 'ArrowRight' ? 1 : items.length - 1)) % items.length].id, true); $('#tab-' + state.cur).focus(); });
    function show(id, user) { state.cur = id; items.forEach((it) => { $('#tab-' + it.id, bar)?.setAttribute('aria-selected', String(it.id === id)); panels[it.id].hidden = it.id !== id; }); const it = items.find((x) => x.id === id);
      if (!state.drawn) state.drawn = {}; if (!state.drawn[id]) state.drawn[id] = it.build(panels[id]) || (() => {}); state.drawn[id](); if (user) { if (onChange) onChange(id); try { history.replaceState(null, '', '#/' + location.hash.replace(/^#\/?/, '').split('/')[0] + '/' + id); } catch (e) { /* ignore */ } } }
    return { el: el('div'), bar, wrap, show, redraw: () => { if (state.cur && state.drawn?.[state.cur]) state.drawn[state.cur](); }, current: () => state.cur, init: () => show(active || items[0].id) }; }

  function fail(host, msg) { host.replaceChildren(el('div', 'fail', msg || 'Failed to load')); }
  return { $, $$, el, F, css, SER, showTip, hideTip, width, svgIn, hbars, cols, line, seg, card, table, csvButton, tabs, fail };
})();
