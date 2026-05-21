# UI restructure for better UX & industry standards

**Status:** implementing
**Date:** 2026-05-21

## Goal

The dashboard currently has 22 templates wired to an 11-item flat top nav with
inline styles on nearly every element. It works but feels "taped together" —
new users have no obvious path through it, and adding a new page means more
nav clutter. This restructure changes the navigation surface, layout shell,
and shared component classes — **without removing a single feature or
changing a single route**. Every URL, form action, and HTMX endpoint stays.

## Non-goals

- No new functionality.
- No backend / API changes.
- No JS build pipeline. HTMX + Chart.js + a small `app.js` is enough.
- No new dependencies.

## New information architecture

Left sidebar with 5 groups (was: flat 11-item top bar):

```
●  forecaster              [▶ Training active / ⏸ Paused]

DASHBOARD
  Overview

FORECASTS                  (consumer-facing: what is being predicted)
  Targets
  Instances

ANALYTICS                  (cross-cutting analysis)
  Trends
  Compare
  Models

TRAINING                   (operations: how/when training runs)
  Runs
  Custom Run
  Schedule

SETTINGS                   (configuration)
  Targets
  Metrics
  Training
  Config
```

Route mapping is 1:1 with today. The Manage subnav (`_manage_subnav.html`)
goes away because the sidebar already shows the Settings children.

## Top bar

A thin sticky bar above the content area, persistent on every page:

- Breadcrumb / page title on the left.
- Refresh stamp ("updated 14:21:03").
- API docs link.

The "training control" big card on the Overview page also stays where it is
— it's a domain-relevant block, not a chrome element. But the **status
chip** (▶ Active / ⏸ Paused) lives in the sidebar header so it's visible
from every page; clicking it goes to the Overview's training card.

## Design system

A new `app.css` introduces tokens + components. All inline `style="…"`
attributes are stripped from templates and replaced with these classes.

### Tokens

```
Spacing:   --space-1=4px  --space-2=8px  --space-3=12px  --space-4=16px
           --space-5=20px --space-6=24px --space-8=32px --space-10=40px
Radius:    --radius-sm=4px --radius=6px --radius-lg=10px
Color:     --bg            (page background)
           --surface       (cards / table body)
           --surface-2     (table head, inputs, hovers)
           --border
           --text          --text-muted   --text-dim
           --primary       (accent: sky blue)
           --primary-fg    (text on primary)
           --accent-2      (purple, B-target & secondary chart)
           --success / --warning / --danger
Shadow:    --shadow-sm --shadow
```

### Components

- `.app-shell` — grid: sidebar + main.
- `.sidebar`, `.sidebar-brand`, `.sidebar-status`, `.sidebar-group`,
  `.sidebar-link`, `.sidebar-toggle` (mobile).
- `.topbar`, `.crumbs`, `.topbar-actions`.
- `.page` — outer wrapper for content with a max-width + padding.
- `.page-header` (flex: title block + right actions).
- `.page-actions` (gap'd row of buttons / forms).
- `.section` (h2 + content wrapper with consistent spacing).
- `.kpi-grid`, `.kpi-card` — replaces ad-hoc `.grid` + `.card`.
- `.toolbar` / `.toolbar-row` — replaces the inconsistent `.filterbar`.
- `.field` (vertical label + input), `.field-inline` (horizontal).
- `.btn`, `.btn-secondary`, `.btn-ghost`, `.btn-danger`, `.btn-sm`.
- `.table` (styled), `.table-zebra`.
- `.pill`, `.pill-good`, `.pill-warn`, `.pill-bad`, `.pill-muted`.
- `.chart-card`, `.chart-card-title`, `.chart-grid-2` (replaces `.two-col`,
  with a saner mobile breakpoint).
- `.code-block`, `.code`.
- `.empty` (centered empty-state).
- `.breadcrumb` (back-link strip on detail pages).
- `.notice`, `.notice-warn`, `.notice-info` (replaces ad-hoc bordered cards).
- `.tabs`, `.tab` (in-page tabs — used on Models to split its 5 charts).

### Mobile

- Sidebar collapses to an icon column at ≤960px, slides over at ≤640px.
- `.chart-grid-2` becomes single-column at ≤880px (was 980px — close enough).
- KPI grid wraps at 180px min-width (unchanged).

## Per-page touches (no feature loss)

- **Overview.** Same blocks, new shell. Training-control card and the three
  HTMX fragments are unchanged.
- **Instances / Targets.** Same tables, new page-header + toolbar.
- **Instance detail / Target detail / Run detail.** Add a `.breadcrumb` row
  at the top. Action buttons move to `.page-actions`.
- **Runs.** "Cancel all active" moves into `.page-actions` (was a separate
  red button taking a whole row). Error groups become a `.section`.
- **Models.** Page header + filter toolbar stay. The 5 charts become 3 tabs
  (`Performance`, `Speed`, `Distribution`). Details table sits below. Same
  data; same filters; less scrolling.
- **Custom Run.** Same form, same fieldsets. Wrapped in the new shell and
  styled via component classes (no inline styles). Saved Configs and Active
  Runs become two `.section`s.
- **Trends / Compare / Schedule.** Same content, new shell. Toolbars use
  `.toolbar`.
- **Manage Targets / Metrics / Training.** Same forms. The `_manage_subnav`
  partial is removed (sidebar covers it). Settings/Targets and Settings/
  Training inherit the same toolbar styling.
- **Config.** Same blocks; the reload button moves into `.page-actions`.

## Files touched

- `src/forecaster/ui/templates/base.html` — full rewrite for the shell.
- `src/forecaster/ui/templates/_sidebar.html` — new partial.
- `src/forecaster/ui/templates/_topbar.html` — new partial.
- `src/forecaster/ui/templates/_breadcrumb.html` — new partial.
- `src/forecaster/ui/templates/_manage_subnav.html` — deleted (replaced by
  sidebar groups).
- `src/forecaster/ui/templates/*.html` — every page template updated to
  remove inline styles and use the new component classes. No content or
  feature changes.
- `src/forecaster/ui/static/css/app.css` — rewritten with tokens and the
  components listed above.
- `src/forecaster/ui/static/js/app.js` — small additions for sidebar
  collapse + tab switcher on Models.
- `src/forecaster/ui/routes.py` — **no changes**. (The `active` context key
  values stay the same; the sidebar reads them.)

## Out of scope (deliberately)

- Light-mode theme.
- Real-time WebSocket updates.
- Reorganising the underlying routes/URLs (would break bookmarks and
  Grafana links).
- New analytics views.

## Rollout

Single PR. The compose stack reloads templates and serves new static
files on next request. No DB or schema changes; no migration; no env-var
changes.
