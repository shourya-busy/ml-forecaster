// Explore page: PromQL builder + raw runner + Chart.js renderer.
// Catalog (instances) is loaded once on page load from /explore/catalog;
// the chart redraws each time the user clicks Run.

(function() {
  const ctx = window.EXPLORE_CTX || {};
  const HISTORY_KEY = "forecaster.explore.history";
  const HISTORY_MAX = 10;
  let chart = null;

  // ----- helpers -----

  const $ = (id) => document.getElementById(id);

  function fmtTs(iso) {
    try {
      return new Intl.DateTimeFormat("en-IN", {
        timeZone: ctx.displayTz || "UTC",
        month: "short", day: "2-digit",
        hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
      }).format(new Date(iso));
    } catch (_) { return iso; }
  }

  function setStatus(msg, kind) {
    const el = $("exploreStatus");
    el.textContent = msg || "";
    el.style.color = kind === "error" ? "var(--danger)" : "var(--text-muted)";
  }

  function showWarnings(warnings) {
    const host = $("exploreWarnings");
    host.innerHTML = "";
    for (const w of warnings || []) {
      const div = document.createElement("div");
      div.className = "notice notice-warn";
      div.style.marginTop = "var(--space-2)";
      div.textContent = w;
      host.appendChild(div);
    }
  }

  // ----- catalog → instance dropdown -----

  async function loadCatalog() {
    try {
      const r = await fetch("/explore/catalog");
      if (!r.ok) throw new Error("HTTP " + r.status);
      const data = await r.json();
      const sel = $("builderInstance");
      for (const inst of data.instances || []) {
        const opt = document.createElement("option");
        opt.value = inst;
        opt.textContent = inst;
        sel.appendChild(opt);
      }
      if (data.discovery_error) {
        setStatus("instance discovery failed — using static list. " + data.discovery_error, "error");
      }
    } catch (err) {
      setStatus("Could not load catalog: " + err.message, "error");
    }
  }

  // ----- builder → PromQL -----

  function buildPromql() {
    const metricSel = $("builderMetric");
    const baseQuery = metricSel.value.trim();
    if (!baseQuery) {
      setStatus("Pick a metric first.", "error");
      return null;
    }
    const instance = $("builderInstance").value.trim();
    const agg = $("builderAgg").value;
    const label = ctx.instanceLabel || "instance";

    // Inject {instance="X"} into the first selector. This is a coarse string
    // op — works for the configured metrics which all start with metric{...}
    // or just metric(...). For complex expressions, switch to Raw mode.
    let q = baseQuery;
    if (instance) {
      const labelFilter = `${label}="${instance}"`;
      if (/\{[^{}]*\}/.test(q)) {
        q = q.replace(/\{([^{}]*)\}/, (_m, inner) => {
          const trimmed = inner.trim();
          return trimmed ? `{${trimmed},${labelFilter}}` : `{${labelFilter}}`;
        });
      } else {
        // Insert {label="…"} immediately after the metric name (first ident).
        q = q.replace(/^([a-zA-Z_:][a-zA-Z0-9_:]*)/, `$1{${labelFilter}}`);
      }
    }
    if (agg !== "raw") {
      q = `${agg} by (${label}) (${q})`;
    }
    return q;
  }

  // ----- range presets -----

  function rangeBounds() {
    const range = $("exploreRange").value;
    const end = new Date();
    if (range === "custom") {
      const startVal = $("exploreStart").value;
      const endVal = $("exploreEnd").value;
      if (!startVal || !endVal) throw new Error("Custom range needs both start and end.");
      return { start: new Date(startVal), end: new Date(endVal) };
    }
    const map = { "1h": 1, "6h": 6, "24h": 24, "7d": 168 };
    const hours = map[range] || 6;
    return { start: new Date(end.getTime() - hours * 3600 * 1000), end };
  }

  // ----- run -----

  async function runQuery() {
    const q = $("exploreQuery").value.trim();
    if (!q) { setStatus("Enter a query (or use the Builder to generate one).", "error"); return; }
    let bounds;
    try { bounds = rangeBounds(); }
    catch (e) { setStatus(e.message, "error"); return; }
    const step = $("exploreStep").value.trim() || "60s";

    setStatus("Running…");
    showWarnings([]);
    const url = new URL("/explore/query", window.location.origin);
    url.searchParams.set("query", q);
    url.searchParams.set("start", bounds.start.toISOString());
    url.searchParams.set("end", bounds.end.toISOString());
    url.searchParams.set("step", step);

    let payload;
    try {
      const r = await fetch(url);
      payload = await r.json();
      if (!r.ok) throw new Error(payload.detail || ("HTTP " + r.status));
    } catch (err) {
      setStatus("Query failed: " + err.message, "error");
      return;
    }

    pushHistory({ query: q, step, range: $("exploreRange").value });
    drawChart(payload);
    showWarnings(payload.warnings || []);
    setStatus(`Returned ${payload.series.length} series in ${payload.request.step} steps.`);
    $("exploreChartTitle").textContent = q;
    $("exploreSeriesCount").textContent =
      payload.series.length + " series · " + payload.request.step + " step";

    // Cache the latest payload so the AI explainer button can build context.
    window._lastExplorePayload = payload;
  }

  // ----- AI explainer context -----

  window.fcAIContext = window.fcAIContext || {};
  window.fcAIContext.explore = function() {
    const payload = window._lastExplorePayload;
    if (!payload || !payload.series || !payload.series.length) return null;
    // Build a virtual "actuals" stream by flattening the first series (or
    // averaging across series if many). The LLM does best with one column,
    // so we send the first series and note how many exist.
    const first = payload.series[0];
    return {
      kind: "explore",
      title: payload.request.query,
      actuals: first.values.map(([ts, value]) => ({ ts, value })),
      forecast: [],
      extra: {
        query: payload.request.query,
        step: payload.request.step,
        series_count: payload.series.length,
        first_instance: first.instance,
      },
    };
  };

  function drawChart(payload) {
    const canvas = $("exploreChart");
    if (!canvas || !window.Chart) return;
    const tsSet = new Set();
    for (const s of payload.series) for (const [t] of s.values) tsSet.add(t);
    const allTs = [...tsSet].sort();
    const tsIdx = new Map(allTs.map((t, i) => [t, i]));
    const datasets = payload.series.map((s) => {
      const data = new Array(allTs.length).fill(null);
      for (const [t, v] of s.values) data[tsIdx.get(t)] = v;
      const color = fc.colors.forAlgo(s.instance);
      return {
        label: s.instance,
        data, borderColor: color, backgroundColor: color,
        borderWidth: 1.5, pointRadius: 0, tension: .2, fill: false, spanGaps: true,
      };
    });
    if (chart) chart.destroy();
    chart = new Chart(canvas, {
      type: "line",
      data: { labels: allTs.map(fmtTs), datasets },
      options: fc.commonChartOptions(),
    });
  }

  // ----- history -----

  function pushHistory(entry) {
    let hist = [];
    try { hist = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); }
    catch (_) { hist = []; }
    hist = hist.filter(h => h.query !== entry.query);
    hist.unshift({ ...entry, at: new Date().toISOString() });
    hist = hist.slice(0, HISTORY_MAX);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(hist));
    renderHistory();
  }

  function renderHistory() {
    const ul = $("exploreHistory");
    if (!ul) return;
    let hist = [];
    try { hist = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); }
    catch (_) {}
    ul.innerHTML = "";
    if (!hist.length) {
      const li = document.createElement("li");
      li.textContent = "(no recent queries yet)";
      ul.appendChild(li);
      return;
    }
    for (const h of hist) {
      const li = document.createElement("li");
      li.style.marginBottom = "4px";
      const a = document.createElement("a");
      a.href = "#";
      a.textContent = h.query;
      a.style.fontFamily = "ui-monospace, Menlo, monospace";
      a.addEventListener("click", (ev) => {
        ev.preventDefault();
        $("exploreQuery").value = h.query;
        if (h.step) $("exploreStep").value = h.step;
        if (h.range) $("exploreRange").value = h.range;
        // Switch to Raw tab for visibility
        document.querySelector('.tab[data-tab="rawTab"]')?.click();
      });
      li.appendChild(a);
      ul.appendChild(li);
    }
  }

  // ----- wire up -----

  document.addEventListener("DOMContentLoaded", () => {
    loadCatalog();
    renderHistory();

    $("builderGenerate")?.addEventListener("click", () => {
      const q = buildPromql();
      if (q) {
        $("exploreQuery").value = q;
        document.querySelector('.tab[data-tab="rawTab"]')?.click();
        setStatus("Generated PromQL — click Run to execute.");
      }
    });
    $("exploreRun")?.addEventListener("click", runQuery);

    $("exploreRange")?.addEventListener("change", (ev) => {
      const showCustom = ev.target.value === "custom";
      for (const el of document.querySelectorAll("[data-custom-only]")) {
        el.hidden = !showCustom;
      }
    });

    // Submit on Ctrl/Cmd+Enter from the textarea for power users.
    $("exploreQuery")?.addEventListener("keydown", (ev) => {
      if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") {
        ev.preventDefault();
        runQuery();
      }
    });
  });
})();
