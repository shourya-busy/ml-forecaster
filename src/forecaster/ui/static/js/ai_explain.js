// AI explainer helper — shared across target_detail, graph tiles and explore.
//
// Page templates render an "Explain (AI)" button with:
//   <button class="btn btn-secondary btn-sm ai-explain-btn"
//           data-ai-context-id="some-global-name"
//           data-ai-context-fn="window.foo">Explain (AI)</button>
// The button reads `window[data-ai-context-fn]()` at click-time to build the
// ExplainRequest body, posts it to /ai/explain, and renders the result into
// the shared modal in base.html.
//
// We probe /ai/status once per page load. If the explainer is disabled or
// the Ollama server is unreachable, every .ai-explain-btn is hidden — no
// dead buttons in the UI.

(function() {
  window.fcAI = window.fcAI || {};

  let probed = null;

  async function probeStatus() {
    if (probed !== null) return probed;
    try {
      const r = await fetch("/ai/status");
      probed = await r.json();
    } catch (_) {
      probed = { enabled: false, reachable: false };
    }
    return probed;
  }

  function applyAvailability(status) {
    const hide = !(status.enabled && status.reachable);
    for (const btn of document.querySelectorAll(".ai-explain-btn")) {
      if (hide) {
        btn.hidden = true;
      } else {
        btn.hidden = false;
        btn.title = `Uses ${status.model || "ollama"} @ ${status.base_url || ""}`;
      }
    }
  }

  function ensureModal() {
    let m = document.getElementById("aiExplainModal");
    if (m) return m;
    m = document.createElement("div");
    m.id = "aiExplainModal";
    m.className = "ai-modal";
    m.hidden = true;
    m.innerHTML = `
      <div class="ai-modal-backdrop" data-close></div>
      <div class="ai-modal-card" role="dialog" aria-modal="true" aria-labelledby="aiModalTitle">
        <div class="ai-modal-head">
          <h3 id="aiModalTitle">Explanation</h3>
          <button type="button" class="btn btn-ghost btn-sm" data-close aria-label="Close">×</button>
        </div>
        <div class="ai-modal-meta sub"></div>
        <div class="ai-modal-body"></div>
        <div class="ai-modal-foot">
          <button type="button" class="btn btn-ghost btn-sm" data-close>Close</button>
        </div>
      </div>`;
    document.body.appendChild(m);
    for (const el of m.querySelectorAll("[data-close]")) {
      el.addEventListener("click", closeModal);
    }
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && !m.hidden) closeModal();
    });
    return m;
  }

  function openModal(title) {
    const m = ensureModal();
    m.querySelector("#aiModalTitle").textContent = title || "Explanation";
    m.querySelector(".ai-modal-meta").textContent = "Asking the model…";
    m.querySelector(".ai-modal-body").innerHTML = '<div class="ai-spinner">⏳</div>';
    m.hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeModal() {
    const m = document.getElementById("aiExplainModal");
    if (m) m.hidden = true;
    document.body.style.overflow = "";
  }

  function renderResult(text, meta) {
    const m = ensureModal();
    m.querySelector(".ai-modal-meta").textContent = meta || "";
    // Render newlines as paragraphs, escape HTML.
    const body = m.querySelector(".ai-modal-body");
    body.innerHTML = "";
    const div = document.createElement("div");
    div.className = "ai-explanation";
    div.textContent = text;
    body.appendChild(div);
  }

  function renderError(msg) {
    const m = ensureModal();
    m.querySelector(".ai-modal-meta").textContent = "Error";
    const body = m.querySelector(".ai-modal-body");
    body.innerHTML = "";
    const div = document.createElement("div");
    div.className = "ai-explanation ai-error";
    div.textContent = msg;
    body.appendChild(div);
  }

  async function runExplain(btn) {
    const fnName = btn.dataset.aiContextFn;
    const titleHint = btn.dataset.aiTitle || "Explanation";
    if (!fnName) { renderError("button missing data-ai-context-fn"); return; }
    const ctxFn = fnName.split(".").reduce((o, k) => (o ? o[k] : undefined), window);
    if (typeof ctxFn !== "function") {
      renderError(`could not find context builder ${fnName}()`);
      return;
    }
    let body;
    try { body = ctxFn(); }
    catch (e) { renderError("could not build context: " + e.message); return; }
    if (!body) { renderError("no chart data to explain yet"); return; }

    openModal(titleHint);
    const started = Date.now();
    try {
      const r = await fetch("/ai/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await r.json();
      if (!r.ok) throw new Error(payload.detail || ("HTTP " + r.status));
      const elapsed = ((Date.now() - started) / 1000).toFixed(1);
      renderResult(payload.explanation, `${payload.model} · ${elapsed}s`);
    } catch (err) {
      renderError(err.message);
    }
  }

  function wireButtons() {
    for (const btn of document.querySelectorAll(".ai-explain-btn")) {
      if (btn.dataset.aiWired === "1") continue;
      btn.dataset.aiWired = "1";
      btn.addEventListener("click", () => runExplain(btn));
    }
  }

  fcAI.refresh = () => { wireButtons(); probeStatus().then(applyAvailability); };

  document.addEventListener("DOMContentLoaded", () => {
    wireButtons();
    probeStatus().then(applyAvailability);
  });

  // HTMX swaps drop in new buttons (e.g., gallery tile fragments) — re-wire.
  document.addEventListener("htmx:afterSwap", () => {
    wireButtons();
    probeStatus().then(applyAvailability);
  });
})();
