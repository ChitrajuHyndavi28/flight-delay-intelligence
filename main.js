/* Entry point: loads the data files, builds the sidebar, routes #/page/tab, handles theme and resize. */
(() => {
  'use strict';
  const { $, $$, el } = L;
  const IDS = ['by_month', 'by_hour', 'by_dow', 'airlines', 'airline_month', 'airports_rank', 'weather_class', 'cause_share', 'cause_month', 'cancel_reasons', 'propagation', 'buffer', 'models', 'pr_curves', 'weather_effect', 'paper_vs_ours', 'tuning',
    'shap', 'gain_groups', 'subgroups', 'ablation', 'clusters', 'airport_map', 'rules'];
  const NAV = [['overview', 'Overview', 'var(--c-overview)', ''], ['sep', 'Prediction moments'], ['t60', 'T-60', 'var(--c-t60)', '1 h before'], ['dayahead', 'Day-ahead', 'var(--c-day)', 'day before'], ['postdep', 'Post-departure', 'var(--c-post)', 'after take-off'],
    ['sep', 'Analysis'], ['evaluation', 'Metric evaluation', 'var(--c-eval)', 'all models'], ['explore', 'Explore data', 'var(--c-explore)', ''], ['about', 'About and limits', 'var(--c-about)', '']];
  const nav = $('#nav'), view = $('#view'), cache = {}; let current = null, currentId = null;
  NAV.forEach(([id, label, color, sub]) => { if (id === 'sep') { nav.append(el('div', 'sep', label)); return; } const a = el('a'); a.href = '#/' + id; a.dataset.page = id; a.style.setProperty('--dot', color); a.append(el('i'), document.createTextNode(label)); if (sub) a.append(el('small', null, sub)); nav.append(a); });

  // theme: system -> light -> dark
  const themes = ['system', 'light', 'dark']; let ti = 0; try { ti = Math.max(0, themes.indexOf(localStorage.getItem('fdi-theme'))); } catch (e) { /* storage unavailable */ }
  const applyTheme = () => { const t = themes[ti]; if (t === 'system') document.documentElement.removeAttribute('data-theme'); else document.documentElement.dataset.theme = t; $('#theme').textContent = 'Theme: ' + t; try { localStorage.setItem('fdi-theme', t); } catch (e) { /* ignore */ } };
  $('#theme').addEventListener('click', () => { ti = (ti + 1) % themes.length; applyTheme(); if (current) current.redraw(); }); applyTheme();

  function route() { const parts = (location.hash.replace(/^#\/?/, '') || 'overview').split('/'), id = Pages.build[parts[0]] ? parts[0] : 'overview', tab = parts[1];
    if (!window.DATA) return; if (!cache[id]) cache[id] = Pages.build[id](); const page = cache[id];
    if (currentId !== id) { view.replaceChildren(page.el); currentId = id; current = page; $$('a', nav).forEach((a) => (a.dataset.page === id ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current'))); scrollTo(0, 0); document.title = (NAV.find((n) => n[0] === id)?.[1] || 'Overview') + ' · Flight Delay Intelligence'; view.style.animation = 'none'; void view.offsetWidth; view.style.animation = ''; }
    page.init(tab); if (page.tabs) { const want = page.tabs.current(); if (parts[1] !== want) history.replaceState(null, '', '#/' + id + '/' + want); } }
  const origShow = null;
  window.addEventListener('hashchange', route);
  let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => current && current.redraw(), 150); });

  Promise.all(IDS.map((id) => fetch('data/' + id + '.json').then((r) => { if (!r.ok) throw new Error(id); return r.json(); }).then((d) => [id, d])))
    .then((e) => { window.DATA = Object.fromEntries(e); route(); })
    .catch((err) => { view.replaceChildren(el('div', 'fail', 'Could not load the result files (' + err.message + '). Start the portal with: bash run_dashboard.sh')); });
  fetch('/api/options').then((r) => r.json()).then((o) => { $('#side-note').textContent = 'Live predictor ready · ' + Object.keys(o.routes).length.toLocaleString() + ' routes'; }).catch(() => { $('#side-note').textContent = 'Predictor offline (static results only)'; });
})();
