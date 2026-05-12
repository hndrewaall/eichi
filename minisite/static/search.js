// eichi minisite — debounced query + morphdom result merge.
//
// The user types in #search-q. Every keystroke schedules a tick after
// DEBOUNCE_MS of inactivity that fires GET /api/search?q=<>&k=<>&source=<>
// and morphdom-merges the returned cards into #results-list. Morphdom
// keys cards by data-result-key (source:path:chunk_idx) so unchanged
// rows are not re-rendered, eliminating flicker as the user adds /
// removes characters.
//
// The form-level controls (k slider + source select) trigger the same
// tick path immediately on change.
//
// Snippet rendering: eichi returns plain text snippets (already
// decoded). We HTML-escape and bold-highlight any token from the query
// that appears in the snippet (case-insensitive, word-boundary).
//
// Result paths render as plain text by default. The card stays
// inert apart from in-app interactions (open in a new tab, expand a
// snippet, etc.) — adding a clickable deep-link surface for a new
// source connector is an out-of-scope extension.

(function () {
  'use strict';
  const DEBOUNCE_MS = 300;
  const ABORT_BUFFER_MS = 50;

  const root            = document.getElementById('search-root');
  const form            = document.getElementById('search-form');
  const qInput          = document.getElementById('search-q');
  const kInput          = document.getElementById('search-k');
  const sourceSelect    = document.getElementById('search-source');
  const yearMinInput    = document.getElementById('search-year-min');
  const yearMaxInput    = document.getElementById('search-year-max');
  const addedSinceSel   = document.getElementById('search-added-since');
  const retrievalSel    = document.getElementById('search-retrieval');
  const resultsList     = document.getElementById('results-list');
  const banner          = document.getElementById('search-banner');
  const countEl         = document.getElementById('result-count');
  const elapsedEl       = document.getElementById('elapsed');
  const announceEl      = document.getElementById('status-announce');

  if (!root || !form || !qInput || !resultsList) return;

  // ---------------------------------------------------------------------
  // Escape helpers — same shape as queue-minisite/refresh.js so the
  // markup-as-string approach stays consistent across apps.
  // ---------------------------------------------------------------------
  function esc(s) {
    if (s === null || s === undefined) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }
  function attr(s) {
    if (s === null || s === undefined) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // ---------------------------------------------------------------------
  // Snippet highlighting — bold-tag any term from the query that appears
  // in the snippet, ASCII-case-insensitive. We escape FIRST then inject
  // <mark> tags so the highlight markup itself doesn't get escaped.
  // ---------------------------------------------------------------------
  function highlight(snippet, query) {
    let html = esc(snippet);
    if (!query) return html;
    // Tokenize on whitespace; ignore short / common stopwords. Cap at
    // 8 tokens to keep the regex bounded.
    const stops = new Set(['the','a','an','of','and','or','to','for','in','on','is','was']);
    const tokens = [...new Set(
      query.toLowerCase()
        .split(/[\s,.\-_/()[\]{}]+/)
        .filter((t) => t.length >= 2 && !stops.has(t))
        .slice(0, 8)
    )];
    if (!tokens.length) return html;
    // Build one alternation regex; escape regex meta-characters.
    const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp('(' + tokens.map(escRe).join('|') + ')', 'gi');
    html = html.replace(re, '<mark>$1</mark>');
    return html;
  }

  // ---------------------------------------------------------------------
  // Source → optional outbound link. The default ships with no outbound
  // links — every result path renders as plain text. Add a per-source
  // case below if you want a particular connector's rows to become
  // clickable (e.g. a public URL derived from the path).
  // ---------------------------------------------------------------------
  function linkFor(result) {
    const p = result.path || '';
    if (!p) return null;
    // No public link in the default build.
    return null;
  }

  // ---------------------------------------------------------------------
  // Relative-age helper. Compact "Nh ago" / "2d ago" / "now" labels
  // matching the eichi CLI shape. Negative values (clock skew, future
  // timestamp) collapse to "in <abs>".
  // ---------------------------------------------------------------------
  function relativeAge(unixSeconds) {
    if (unixSeconds === null || unixSeconds === undefined) return '';
    const now = Date.now() / 1000;
    let s = now - Number(unixSeconds);
    if (!isFinite(s)) return '';
    const sign = (s < 0) ? 'in ' : '';
    s = Math.abs(s);
    if (s < 60) return sign ? 'in <1m' : 'now';
    const units = [
      [60 * 60 * 24 * 365, 'y'],
      [60 * 60 * 24 * 30,  'mo'],
      [60 * 60 * 24 * 7,   'w'],
      [60 * 60 * 24,       'd'],
      [60 * 60,            'h'],
      [60,                 'm'],
    ];
    for (const [size, label] of units) {
      if (s >= size) {
        const n = Math.floor(s / size);
        return sign ? `in ${n}${label}` : `${n}${label} ago`;
      }
    }
    return '';
  }

  // ---------------------------------------------------------------------
  // Same-day check — both unix epochs interpreted in the viewer's local
  // timezone (matches LocalTime rendering). Used to decide whether a
  // cluster row's mtime_end suffix renders as "–HH:MM" (same day) vs
  // "–YYYY-MM-DD HH:MM" (cross-day).
  // ---------------------------------------------------------------------
  function sameLocalDay(unixA, unixB) {
    if (!unixA || !unixB) return false;
    const a = new Date(Number(unixA) * 1000);
    const b = new Date(Number(unixB) * 1000);
    return (
      a.getFullYear() === b.getFullYear() &&
      a.getMonth() === b.getMonth() &&
      a.getDate() === b.getDate()
    );
  }

  // ---------------------------------------------------------------------
  // Build the timestamp <span> markup. Renders as "YYYY-MM-DD HH:MM
  // (3h ago)" in the viewer's local timezone via LocalTime.hydrate()
  // (see local-time.js). data-ts-unix is what relativeAge reads.
  // ts_kind annotates the tooltip ("upstream timestamp" vs "indexed at").
  //
  // For cluster rows where mtime_end > mtime AND both fall on the same
  // local day, render the absolute portion as a range:
  //     "13:41–13:48 ET (3h ago)"
  // For cross-day cluster ranges OR single-message clusters
  // (mtime_end == 0 or == mtime), render as today (no suffix).
  // ---------------------------------------------------------------------
  function tsHtml(result) {
    const iso = result.ts_iso;
    const tsUnix = result.ts;
    if (!iso || !tsUnix) return '';
    const kind = result.ts_kind || '';
    const tipBase = (kind === 'mtime')
      ? 'Upstream timestamp (per-connector relevance time)'
      : (kind === 'indexed')
        ? 'When eichi ingested this row'
        : 'Timestamp';
    const age = relativeAge(tsUnix);
    const ageHtml = age ? ` <span class="result-ts-age">(${esc(age)})</span>` : '';

    // Cluster end-time range. Renders only when (a) the row is a cluster,
    // (b) mtime_end is later than the picked ts, and (c) both are on the
    // same local day. local-time.js handles the time formatting via the
    // `time` format key; same trick as the absolute stamp above.
    const isCluster = (result.kind === 'cluster');
    const mtimeEndIso = result.mtime_end_iso || '';
    const mtimeEndUnix = Number(result.mtime_end || 0);
    let endHtml = '';
    if (isCluster && mtimeEndIso && mtimeEndUnix > Number(tsUnix) + 1 &&
        sameLocalDay(tsUnix, mtimeEndUnix)) {
      endHtml =
        `<span class="result-ts-end-sep">–</span>` +
        `<span class="result-ts-end" ` +
          `data-local-time-iso="${attr(mtimeEndIso)}" ` +
          `data-local-time-fmt="time-short">` +
          esc(mtimeEndIso) +
        `</span>`;
    }

    // local-time.js will fill the textContent via [data-local-time-iso].
    // Pre-fill the ISO so the page is meaningful even if the helper
    // hasn't run yet (no FOUC of empty text).
    return (
      `<span class="result-ts" title="${attr(tipBase)}">` +
        `<span class="result-ts-stamp" ` +
          `data-local-time-iso="${attr(iso)}" ` +
          `data-local-time-fmt="datetime" ` +
          `data-local-time-tooltip>` +
          esc(iso) +
        `</span>` +
        endHtml +
        ageHtml +
      `</span>`
    );
  }

  // ---------------------------------------------------------------------
  // Score / band rendering. The backend ships:
  //   * `score`          — raw vec0 L2 distance (smaller = closer; opaque
  //                        in isolation, preserved for backwards-compat).
  //   * `similarity`     — 0–1 projection of the same distance (1 = exact,
  //                        0 = no relationship). What we display as the
  //                        primary number.
  //   * `relevance_band` — named bucket: strong / moderate / weak /
  //                        distant. Drives the colored chip + tooltip.
  // The chip is suppressed when the band field is missing (older worker
  // rolling-upgrade) — the bare similarity number is still meaningful.
  // ---------------------------------------------------------------------
  const BAND_TOOLTIPS = {
    strong:   'Strong match (close vector distance)',
    moderate: 'Moderate match',
    weak:     'Weak match',
    distant:  'Distant — likely no real relationship',
  };

  function scoreHtml(result) {
    const sim = result.similarity;
    const raw = result.score;
    const band = result.relevance_band || '';
    const simStr = (typeof sim === 'number') ? sim.toFixed(2)
                  : (typeof raw === 'number') ? raw.toFixed(3)  // legacy fallback
                  : '—';
    const tipBand = BAND_TOOLTIPS[band] || '';
    const rawTip = (typeof raw === 'number')
      ? `Raw vec0 distance: ${raw.toFixed(3)} (smaller = closer; THRESHOLD = 1.4)`
      : '';
    const tip = [
      'Similarity 0–1 (1 = exact match, 0 = no relationship).',
      tipBand,
      rawTip,
    ].filter(Boolean).join('\n');
    let bandChipHtml = '';
    if (band) {
      bandChipHtml =
        `<span class="badge result-band-chip result-band-${attr(band)}" ` +
          `title="${attr(BAND_TOOLTIPS[band] || band)}">` +
          esc(band) +
        `</span>`;
    }
    return (
      `<span class="result-score" title="${attr(tip)}">${esc(simStr)}</span>` +
      bandChipHtml
    );
  }

  // ---------------------------------------------------------------------
  // Deep-link button stub. The default build never emits an
  // ``app_link`` / ``app_label`` in the JSON envelope, so this always
  // returns an empty string. Kept as an extension point so downstream
  // forks can wire connector-specific "Open in <App>" pills without
  // rewriting the card renderer.
  // ---------------------------------------------------------------------
  function appLinkHtml(result) {
    const url = result.app_link;
    const label = result.app_label;
    if (!url || !label) return '';
    return (
      `<a class="result-app-link" ` +
        `href="${attr(url)}" target="_blank" rel="noopener noreferrer" ` +
        `title="${attr('Open this result in ' + label)}">` +
        `Open in ${esc(label)}` +
      `</a>`
    );
  }

  // ---------------------------------------------------------------------
  // Result card renderer.
  // ---------------------------------------------------------------------
  function renderResultCard(result, query) {
    const source = result.source || 'unknown';
    const path = result.path || '';
    const link = linkFor(result);
    const snippetHtml = highlight(result.snippet || '', query);
    const key = result.key || (source + ':' + path + ':' + (result.chunk_idx ?? ''));

    let pathHtml;
    if (link) {
      pathHtml =
        `<a class="result-path" href="${attr(link)}" target="_blank" rel="noopener noreferrer">${esc(path)}</a>`;
    } else {
      // Plain text — public-facing UI MUST NOT leak private paths as
      // clickable links. Show the path so a logged-in admin can grep
      // locally, but don't make it interactive.
      pathHtml = `<span class="result-path">${esc(path)}</span>`;
    }

    // chunk hint (if eichi reports one)
    let chunkHtml = '';
    if (result.chunk_idx !== null && result.chunk_idx !== undefined && result.chunk_idx !== 0) {
      chunkHtml = `<span class="result-chunk" title="chunk index">#${esc(result.chunk_idx)}</span>`;
    }

    const tsMarkup = tsHtml(result);

    // Cluster badge — mirrors the CLI's "[<source> cluster, N msgs]"
    // prefix. Only rendered for kind="cluster" rows; msg / non-cluster
    // rows omit it entirely so the existing layout is unchanged.
    let clusterBadgeHtml = '';
    if (result.kind === 'cluster') {
      const n = Number(result.cluster_size || 0);
      const label = `cluster, ${n} msg${n === 1 ? '' : 's'}`;
      clusterBadgeHtml =
        `<span class="badge result-cluster-badge" ` +
          `title="cluster of ${esc(String(n))} message${n === 1 ? '' : 's'}">` +
          esc(label) +
        `</span>`;
    }

    const appLinkMarkup = appLinkHtml(result);

    // Path row composes the path text + optional "Open in <App>" pill.
    // Wrapping in a row lets us push the pill to the right edge via CSS
    // flex without reflowing the path-truncation rules.
    const pathRowHtml = appLinkMarkup
      ? `<div class="result-path-row">${pathHtml}${appLinkMarkup}</div>`
      : pathHtml;

    // When the result has a deep-link target, stamp it on the card so a
    // click anywhere in the card (outside another interactive element)
    // navigates there. This widens the click target from the small "Open
    // in <App>" pill (which most users miss) to the entire row
    // (q-2026-05-04-26d6).
    const cardLink = result.app_link || '';
    const cardLinkAttrs = cardLink
      ? ` data-app-link="${attr(cardLink)}" tabindex="0" role="link"`
      : '';
    return (
      `<article class="result-card" data-result-key="${attr(key)}" data-source="${attr(source)}"` +
        (result.kind ? ` data-kind="${attr(result.kind)}"` : '') +
        (result.relevance_band ? ` data-band="${attr(result.relevance_band)}"` : '') +
        cardLinkAttrs +
      `>` +
        `<header class="result-head">` +
          `<span class="badge source-badge source-${attr(source)}">${esc(source)}</span>` +
          clusterBadgeHtml +
          scoreHtml(result) +
          chunkHtml +
          tsMarkup +
        `</header>` +
        pathRowHtml +
        `<p class="result-snippet">${snippetHtml}</p>` +
      `</article>`
    );
  }

  function renderEmpty(query) {
    if (!query) {
      return `<div class="empty-card" id="results-empty">Type a query above to search.</div>`;
    }
    return `<div class="empty-card" id="results-empty">No results for <code>${esc(query)}</code>.</div>`;
  }

  function renderResultsList(results, query) {
    if (!results || !results.length) {
      return `<div class="results-list" id="results-list">${renderEmpty(query)}</div>`;
    }
    const cards = results.map((r) => renderResultCard(r, query)).join('');
    return `<div class="results-list" id="results-list">${cards}</div>`;
  }

  // ---------------------------------------------------------------------
  // Morphdom merge.
  // ---------------------------------------------------------------------
  function mergeResultsDOM(results, query) {
    const html = renderResultsList(results, query);
    const tpl = document.createElement('template');
    tpl.innerHTML = html;
    const newRoot = tpl.content.firstElementChild;
    if (!newRoot) return;
    if (window.morphdom) {
      window.morphdom(resultsList, newRoot, {
        getNodeKey(node) {
          if (node.nodeType !== 1) return undefined;
          if (node.id) return node.id;
          if (node.getAttribute) {
            const k = node.getAttribute('data-result-key');
            if (k) return 'rk:' + k;
          }
          return undefined;
        },
      });
    } else {
      // Fallback (no morphdom): naive innerHTML swap. Acceptable; we
      // only hit this if the vendored asset failed to load.
      resultsList.innerHTML = newRoot.innerHTML;
    }
    // After the DOM merge, render any [data-local-time-iso] timestamps
    // in the viewer's local timezone. LocalTime.hydrate is idempotent —
    // re-running on already-rendered nodes is a no-op unless the iso
    // string changed.
    if (window.LocalTime && typeof window.LocalTime.hydrate === 'function') {
      window.LocalTime.hydrate(resultsList);
    }
  }

  // ---------------------------------------------------------------------
  // Topbar updates.
  // ---------------------------------------------------------------------
  function setTopbar({ count, elapsedMs, error, k, source }) {
    if (countEl) {
      const baseCount = (count === null || count === undefined) ? '—' : `${count}`;
      let label = `${baseCount} result${count === 1 ? '' : 's'}`;
      if (typeof k === 'number' && count !== undefined && count !== null) {
        label += ` (k=${k})`;
      }
      if (source) {
        label += ` · ${source}`;
      }
      countEl.textContent = label;
      countEl.classList.toggle('count-err', !!error);
    }
    if (elapsedEl) {
      if (elapsedMs === null || elapsedMs === undefined) {
        elapsedEl.textContent = '—';
      } else {
        elapsedEl.textContent = `${elapsedMs} ms`;
      }
    }
  }

  function setError(message) {
    if (!banner) return;
    if (message) {
      banner.textContent = message;
      banner.hidden = false;
    } else {
      banner.textContent = '';
      banner.hidden = true;
    }
  }

  function announce(message) {
    if (!announceEl) return;
    announceEl.textContent = message;
  }

  // ---------------------------------------------------------------------
  // Fetch tick.
  // ---------------------------------------------------------------------
  let inflight = null;       // AbortController of the current fetch
  let debounceTimer = null;  // setTimeout handle
  let lastIssuedQuery = null; // last query value we actually fetched

  function readYearInput(el) {
    if (!el) return '';
    const v = (el.value || '').trim();
    if (!v) return '';
    const n = parseInt(v, 10);
    if (!Number.isFinite(n)) return '';
    return String(n);
  }

  function updateLocationBar(params) {
    // Mirror the active filter state in the address bar so deep-linking
    // / share / refresh round-trips. We use `replaceState` (no history
    // entry per keystroke) — the user can still hit back to the page
    // they came from.
    try {
      const url = new URL(window.location.href);
      ['q', 'k', 'source', 'year_min', 'year_max', 'added_since', 'retrieval']
        .forEach((key) => {
          const v = params[key];
          if (v === null || v === undefined || v === '') {
            url.searchParams.delete(key);
          } else {
            url.searchParams.set(key, String(v));
          }
        });
      window.history.replaceState(null, '', url.toString());
    } catch (_) { /* ignore */ }
  }

  async function runQuery() {
    const q = (qInput.value || '').trim();
    const k = Math.max(1, Math.min(parseInt(kInput.value, 10) || 20, 100));
    const source = sourceSelect ? (sourceSelect.value || '') : '';
    const yearMin = readYearInput(yearMinInput);
    const yearMax = readYearInput(yearMaxInput);
    const addedSince = addedSinceSel ? (addedSinceSel.value || '') : '';
    const retrieval = retrievalSel ? (retrievalSel.value || '') : '';

    if (inflight) {
      try { inflight.abort(); } catch (_) { /* ignore */ }
    }
    if (!q) {
      setError(null);
      setTopbar({ count: 0, elapsedMs: 0, error: null, k, source });
      mergeResultsDOM([], '');
      announce('Empty query.');
      lastIssuedQuery = '';
      updateLocationBar({
        q: '', k: String(k), source, year_min: yearMin, year_max: yearMax,
        added_since: addedSince, retrieval,
      });
      return;
    }

    const ctl = new AbortController();
    inflight = ctl;
    lastIssuedQuery = q;

    const url = new URL('/api/search', window.location.origin);
    url.searchParams.set('q', q);
    url.searchParams.set('k', String(k));
    if (source) url.searchParams.set('source', source);
    if (yearMin) url.searchParams.set('year_min', yearMin);
    if (yearMax) url.searchParams.set('year_max', yearMax);
    if (addedSince) url.searchParams.set('added_since', addedSince);
    if (retrieval && retrieval !== 'hybrid') url.searchParams.set('retrieval', retrieval);

    updateLocationBar({
      q, k: String(k), source, year_min: yearMin, year_max: yearMax,
      added_since: addedSince,
      // Skip retrieval=hybrid in the bar (it's the default) so the URL
      // stays compact for the common case.
      retrieval: (retrieval && retrieval !== 'hybrid') ? retrieval : '',
    });

    try {
      const resp = await fetch(url, { signal: ctl.signal, cache: 'no-store' });
      const json = await resp.json();
      if (ctl !== inflight) return; // a newer query landed first
      if (!resp.ok || !json || json.ok === false) {
        const msg = (json && json.error) || `HTTP ${resp.status}`;
        setError(`Search failed: ${msg}`);
        setTopbar({ count: null, elapsedMs: null, error: msg, k, source });
        mergeResultsDOM([], q);
        announce(`Search failed: ${msg}`);
        return;
      }
      const results = json.results || [];
      setError(null);
      setTopbar({
        count: results.length,
        elapsedMs: json.elapsed_ms,
        error: null,
        k: json.k,
        source: json.source,
      });
      mergeResultsDOM(results, q);
      announce(`${results.length} result${results.length === 1 ? '' : 's'} for "${q}".`);
    } catch (e) {
      if (e && e.name === 'AbortError') return;
      const msg = (e && e.message) || String(e);
      setError(`Network error: ${msg}`);
      announce(`Search error: ${msg}`);
    } finally {
      if (ctl === inflight) inflight = null;
    }
  }

  function scheduleQuery() {
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    debounceTimer = setTimeout(() => {
      debounceTimer = null;
      runQuery();
    }, DEBOUNCE_MS);
  }

  function immediateQuery() {
    if (debounceTimer) {
      clearTimeout(debounceTimer);
      debounceTimer = null;
    }
    runQuery();
  }

  // ---------------------------------------------------------------------
  // Card-level deep-link navigation. When a result card has a backend-
  // computed `data-app-link`, treat the entire card as a link: clicking
  // anywhere on it (or pressing Enter/Space when focused) navigates to
  // that URL in a new tab.
  //
  // We bail out when the click landed on an actual <a>, <button>, or
  // form control inside the card so the existing "Open in <App>" pill,
  // path link, and any future inline interactive elements still behave
  // as native widgets (e.g. middle-click, ctrl-click open in new tab,
  // copy link address). We also bail on text selection (mouseup after a
  // drag) so users can highlight snippets without accidentally
  // navigating away.
  //
  // Background: pre-q-2026-05-04-26d6 the card had only a small pill
  // anchor. Andrew (and most users) instinctively clicked the result
  // body / path text and got no navigation — the path was rendered as
  // a plain <span> for sources without a per-source URL builder. The
  // backend has computed the right deep link all along; this hooks it
  // up to a click target that matches user expectation.
  // ---------------------------------------------------------------------
  function findCardWithLink(node) {
    // Walk up from the click target until we either find a result-card
    // with data-app-link or hit the resultsList container.
    let el = node;
    while (el && el !== resultsList) {
      if (el.nodeType === 1) {
        // If we hit an interactive element first, the card-level
        // handler should NOT swallow the click — let the native widget
        // handle it (e.g. the "Open in <App>" pill or a future button).
        const tag = el.tagName;
        if (tag === 'A' || tag === 'BUTTON' || tag === 'INPUT' ||
            tag === 'SELECT' || tag === 'TEXTAREA' || tag === 'LABEL') {
          return null;
        }
        if (el.classList && el.classList.contains('result-card')) {
          const link = el.getAttribute('data-app-link');
          return link ? { card: el, link } : null;
        }
      }
      el = el.parentNode;
    }
    return null;
  }

  resultsList.addEventListener('click', (e) => {
    // Skip if a modifier key is held — the user is clearly trying to do
    // something other than primary-click navigate (cmd/ctrl-click for
    // new tab is delivered by the browser to the focused link;
    // shift-click extends a selection; alt-click downloads). Without
    // this guard a cmd-click on the card body would navigate the
    // current tab, which is the opposite of what cmd-click means.
    if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) {
      return;
    }
    // Don't hijack a text-selection drag.
    const sel = window.getSelection && window.getSelection();
    if (sel && sel.toString && sel.toString().length > 0) {
      return;
    }
    const hit = findCardWithLink(e.target);
    if (!hit) return;
    e.preventDefault();
    // Open in a new tab to match the existing "Open in <App>" pill's
    // target=_blank behavior — keeps the search tab around so the user
    // can keep clicking through hits.
    window.open(hit.link, '_blank', 'noopener');
  });

  resultsList.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const hit = findCardWithLink(e.target);
    if (!hit) return;
    e.preventDefault();
    window.open(hit.link, '_blank', 'noopener');
  });

  // ---------------------------------------------------------------------
  // Wire events.
  // ---------------------------------------------------------------------
  qInput.addEventListener('input', scheduleQuery);
  qInput.addEventListener('search', immediateQuery); // <input type="search"> clear-x
  qInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      immediateQuery();
    }
  });
  if (kInput) kInput.addEventListener('change', immediateQuery);
  if (sourceSelect) sourceSelect.addEventListener('change', immediateQuery);
  // Year inputs: debounce on `input` (typing 1-9-9-9 shouldn't fire 4
  // queries), fire immediately on `change` / blur. Empty value = no
  // filter — handled by readYearInput.
  if (yearMinInput) {
    yearMinInput.addEventListener('input', scheduleQuery);
    yearMinInput.addEventListener('change', immediateQuery);
  }
  if (yearMaxInput) {
    yearMaxInput.addEventListener('input', scheduleQuery);
    yearMaxInput.addEventListener('change', immediateQuery);
  }
  if (addedSinceSel) addedSinceSel.addEventListener('change', immediateQuery);
  if (retrievalSel) retrievalSel.addEventListener('change', immediateQuery);

  // Bootstrap: hydrate filter form from URL params, then auto-fire if
  // there's a query. Useful for deep-links / share URLs that carry
  // every facet (?q=...&year_min=2020&added_since=30d&source=memory).
  const urlParams = new URL(window.location.href).searchParams;
  const initialQ = urlParams.get('q') || '';
  const initialK = urlParams.get('k');
  const initialSource = urlParams.get('source') || '';
  const initialYearMin = urlParams.get('year_min') || '';
  const initialYearMax = urlParams.get('year_max') || '';
  const initialAddedSince = urlParams.get('added_since') || '';
  const initialRetrieval = urlParams.get('retrieval') || '';

  function _setIfOption(sel, value) {
    if (!sel || !value) return;
    for (const opt of sel.options) {
      if (opt.value === value) { sel.value = value; return; }
    }
  }

  if (kInput && initialK && /^\d+$/.test(initialK)) {
    kInput.value = initialK;
  }
  _setIfOption(sourceSelect, initialSource);
  if (yearMinInput && initialYearMin) yearMinInput.value = initialYearMin;
  if (yearMaxInput && initialYearMax) yearMaxInput.value = initialYearMax;
  _setIfOption(addedSinceSel, initialAddedSince);
  _setIfOption(retrievalSel, initialRetrieval);

  if (initialQ) {
    qInput.value = initialQ;
    immediateQuery();
  } else {
    setTopbar({ count: 0, elapsedMs: 0, error: null, k: parseInt(kInput.value, 10) || 20, source: '' });
  }

  // Test hook (mirrors queue-minisite refresh.js export pattern).
  window.__searchUI = {
    runQuery,
    scheduleQuery,
    mergeResultsDOM,
    renderResultCard,
    highlight,
    linkFor,
    appLinkHtml,
    relativeAge,
    tsHtml,
    scoreHtml,
    sameLocalDay,
    BAND_TOOLTIPS,
    DEBOUNCE_MS,
  };
})();
