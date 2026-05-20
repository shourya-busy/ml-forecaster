// Small helpers shared across dashboard pages.

window.fc = window.fc || {};

fc.fmt = {
  num: (v, dp = 3) => {
    if (v === null || v === undefined || isNaN(v)) return "—";
    const n = Number(v);
    if (Math.abs(n) >= 1000) return n.toFixed(0);
    return n.toFixed(dp);
  },
  ts: (iso) => {
    if (!iso) return "—";
    const d = new Date(iso);
    return d.toLocaleString();
  },
  ago: (iso) => {
    if (!iso) return "—";
    const ms = Date.now() - new Date(iso).getTime();
    if (ms < 0) return "in future";
    const s = Math.floor(ms / 1000);
    if (s < 60) return s + "s ago";
    const m = Math.floor(s / 60);
    if (m < 60) return m + "m ago";
    const h = Math.floor(m / 60);
    if (h < 24) return h + "h ago";
    const d = Math.floor(h / 24);
    return d + "d ago";
  },
};

fc.refreshStamp = () => {
  const el = document.querySelector(".refresh-stamp");
  if (el) el.textContent = "updated " + new Date().toLocaleTimeString();
};

document.addEventListener("DOMContentLoaded", fc.refreshStamp);
document.addEventListener("htmx:afterSwap", fc.refreshStamp);

// Color palette for charts — keep in sync across plots so a given algo
// has a consistent line across pages.
fc.colors = {
  byIndex: ["#38bdf8", "#a78bfa", "#22c55e", "#f59e0b", "#f472b6",
            "#10b981", "#fb7185", "#facc15", "#60a5fa", "#34d399"],
  forAlgo: (() => {
    const map = new Map();
    return (algo) => {
      if (!map.has(algo)) {
        map.set(algo, fc.colors.byIndex[map.size % fc.colors.byIndex.length]);
      }
      return map.get(algo);
    };
  })(),
};

fc.commonChartOptions = (yLabel) => ({
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: "nearest", intersect: false },
  scales: {
    x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,.05)" } },
    y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,.05)" },
         title: { display: !!yLabel, text: yLabel || "", color: "#94a3b8" } },
  },
  plugins: {
    legend: { labels: { color: "#cbd5e1" } },
    tooltip: { backgroundColor: "#0b1220", borderColor: "#334155", borderWidth: 1 },
  },
});
