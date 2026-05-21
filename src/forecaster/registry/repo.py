"""Repository helpers used by API + scheduler.

Encapsulates the SQL we don't want scattered across routes/tasks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Base,
    Forecast,
    ModelArtifact,
    Ranking,
    RunMetric,
    SettingsOverride,
    TargetOverride,
    TrainingRun,
)


class RegistryRepo:
    def __init__(self, database_url: str):
        self.engine: Engine = create_engine(database_url, pool_pre_ping=True, future=True)
        self._Session = sessionmaker(self.engine, expire_on_commit=False, future=True)

    def create_schema(self) -> None:
        """Convenience for tests; real deployments use alembic."""
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self):
        s: Session = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ----- writes -----

    def create_run(self, *, instance: str, metric: str, horizon: str, config_snapshot: dict[str, Any]) -> int:
        with self.session() as s:
            run = TrainingRun(
                instance=instance,
                metric=metric,
                horizon=horizon,
                status="running",
                config_snapshot=config_snapshot,
            )
            s.add(run)
            s.flush()
            return run.id

    def mark_completed(self, run_id: int, duration_seconds: float, error: str | None = None) -> None:
        with self.session() as s:
            run = s.get(TrainingRun, run_id)
            if not run:
                return
            run.status = "failed" if error else "completed"
            run.duration_seconds = duration_seconds
            run.completed_at = datetime.now(timezone.utc)
            run.error = error

    def add_metrics(self, run_id: int, algo: str, scores: dict[str, float], *, fold: int = -1) -> None:
        with self.session() as s:
            for k, v in scores.items():
                s.add(RunMetric(run_id=run_id, algo=algo, score_metric=k, value=v, fold=fold))

    def add_artifact(self, run_id: int, algo: str, path: str, size_bytes: int, train_duration_seconds: float) -> None:
        with self.session() as s:
            s.add(ModelArtifact(
                run_id=run_id, algo=algo, path=path,
                size_bytes=size_bytes, train_duration_seconds=train_duration_seconds,
            ))

    def add_forecasts(
        self,
        *,
        run_id: int,
        instance: str,
        metric: str,
        horizon: str,
        algo: str,
        is_best: bool,
        timestamps: Sequence[datetime],
        point: Sequence[float],
        lower: Sequence[float | None],
        upper: Sequence[float | None],
    ) -> None:
        with self.session() as s:
            for ts, p, lo, hi in zip(timestamps, point, lower, upper, strict=True):
                s.add(Forecast(
                    run_id=run_id, instance=instance, metric=metric, horizon=horizon,
                    algo=algo, ts=ts, point=float(p),
                    lower=None if lo is None else float(lo),
                    upper=None if hi is None else float(hi),
                    is_best=is_best,
                ))

    def add_ranking(self, *, run_id: int, instance: str, metric: str, horizon: str, winning_algo: str, ranked: list[dict]) -> None:
        with self.session() as s:
            s.add(Ranking(
                run_id=run_id, instance=instance, metric=metric, horizon=horizon,
                winning_algo=winning_algo, ranked=ranked,
            ))

    # ----- reads -----

    def get_run(self, run_id: int) -> TrainingRun | None:
        with self.session() as s:
            return s.get(TrainingRun, run_id)

    def list_runs(self, *, instance: str | None = None, metric: str | None = None, horizon: str | None = None, limit: int = 50) -> list[TrainingRun]:
        with self.session() as s:
            q = select(TrainingRun).order_by(TrainingRun.started_at.desc()).limit(limit)
            if instance:
                q = q.where(TrainingRun.instance == instance)
            if metric:
                q = q.where(TrainingRun.metric == metric)
            if horizon:
                q = q.where(TrainingRun.horizon == horizon)
            return list(s.scalars(q))

    def latest_forecasts(
        self,
        *,
        instance: str | None = None,
        metric: str | None = None,
        horizon: str | None = None,
        only_best: bool = True,
        algo: str | None = None,
        fresh_since: datetime | None = None,
    ) -> list[Forecast]:
        """Return forecasts whose run is the latest completed for each
        (instance, metric, horizon). Used by API + Prom exposition.

        If `fresh_since` is given, only returns forecasts whose underlying
        TrainingRun.completed_at is >= fresh_since — i.e. drops stale
        forecasts whose run hasn't refreshed lately.
        """
        with self.session() as s:
            # Latest completed run id per target.
            base_q = (
                select(
                    TrainingRun.instance, TrainingRun.metric, TrainingRun.horizon,
                    TrainingRun.id.label("run_id"),
                )
                .where(TrainingRun.status == "completed")
            )
            if fresh_since is not None:
                base_q = base_q.where(TrainingRun.completed_at >= fresh_since)
            subq = (
                base_q
                .order_by(
                    TrainingRun.instance, TrainingRun.metric, TrainingRun.horizon,
                    TrainingRun.completed_at.desc(),
                )
                .distinct(TrainingRun.instance, TrainingRun.metric, TrainingRun.horizon)
                .subquery()
            )
            q = select(Forecast).join(subq, Forecast.run_id == subq.c.run_id)
            if instance:
                q = q.where(Forecast.instance == instance)
            if metric:
                q = q.where(Forecast.metric == metric)
            if horizon:
                q = q.where(Forecast.horizon == horizon)
            if only_best:
                q = q.where(Forecast.is_best.is_(True))
            if algo:
                q = q.where(Forecast.algo == algo)
            return list(s.scalars(q))

    def latest_rankings(
        self, *, instance: str | None = None, metric: str | None = None, horizon: str | None = None
    ) -> list[Ranking]:
        with self.session() as s:
            subq = (
                select(
                    TrainingRun.instance, TrainingRun.metric, TrainingRun.horizon,
                    TrainingRun.id.label("run_id"),
                )
                .where(TrainingRun.status == "completed")
                .order_by(
                    TrainingRun.instance, TrainingRun.metric, TrainingRun.horizon,
                    TrainingRun.completed_at.desc(),
                )
                .distinct(TrainingRun.instance, TrainingRun.metric, TrainingRun.horizon)
                .subquery()
            )
            q = select(Ranking).join(subq, Ranking.run_id == subq.c.run_id)
            if instance:
                q = q.where(Ranking.instance == instance)
            if metric:
                q = q.where(Ranking.metric == metric)
            if horizon:
                q = q.where(Ranking.horizon == horizon)
            return list(s.scalars(q))

    def prune_old_artifacts(self, keep_per_target: int, instances: Iterable[str] | None = None) -> int:
        """Delete artifact rows + return count for outside-of-DB file cleanup."""
        # Implemented lazily; v1 callers can skip and let the volume grow.
        return 0

    # ----- diagnostics -----

    def score_history(
        self,
        *,
        instance: str,
        metric: str,
        horizon: str,
        algo: str | None = None,
        score: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Per-run averaged-fold scores, oldest-first, up to `limit` runs.

        Returns a flat list: [{run_id, completed_at, algo, score, value}, ...].
        """
        with self.session() as s:
            run_q = (
                select(TrainingRun.id, TrainingRun.completed_at)
                .where(TrainingRun.status == "completed")
                .where(TrainingRun.instance == instance)
                .where(TrainingRun.metric == metric)
                .where(TrainingRun.horizon == horizon)
                .order_by(TrainingRun.completed_at.desc())
                .limit(limit)
            )
            run_rows = list(s.execute(run_q))
            if not run_rows:
                return []
            run_ids = [r.id for r in run_rows]
            ts_by_run = {r.id: r.completed_at for r in run_rows}

            m_q = (
                select(RunMetric)
                .where(RunMetric.run_id.in_(run_ids))
                .where(RunMetric.fold == -1)
            )
            if algo:
                m_q = m_q.where(RunMetric.algo == algo)
            if score:
                m_q = m_q.where(RunMetric.score_metric == score)
            metrics = list(s.scalars(m_q))

        out = [
            {
                "run_id": m.run_id,
                "completed_at": ts_by_run[m.run_id].isoformat() if ts_by_run[m.run_id] else None,
                "algo": m.algo,
                "score": m.score_metric,
                "value": m.value,
            }
            for m in metrics
        ]
        out.sort(key=lambda r: (r["completed_at"] or "", r["algo"], r["score"]))
        return out

    def winner_history(
        self,
        *,
        instance: str,
        metric: str,
        horizon: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Per-run winner + composite, oldest-first."""
        with self.session() as s:
            q = (
                select(Ranking, TrainingRun.completed_at)
                .join(TrainingRun, TrainingRun.id == Ranking.run_id)
                .where(TrainingRun.status == "completed")
                .where(Ranking.instance == instance)
                .where(Ranking.metric == metric)
                .where(Ranking.horizon == horizon)
                .order_by(TrainingRun.completed_at.desc())
                .limit(limit)
            )
            rows = list(s.execute(q))
        out = [
            {
                "run_id": r.Ranking.run_id,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "winning_algo": r.Ranking.winning_algo,
                "ranked": r.Ranking.ranked,
            }
            for r in rows
        ]
        out.sort(key=lambda x: x["completed_at"] or "")
        return out

    def winners_summary(self, recent_window: int = 10) -> list[dict[str, Any]]:
        """One row per (instance, metric, horizon).

        Each row reports the current winner, the previous winner (or None),
        the timestamp the current streak started, and the number of distinct
        winners observed across the most recent `recent_window` runs.
        """
        with self.session() as s:
            # All completed rankings, newest first; we'll group in Python so
            # the SQL stays portable across SQLite + Postgres.
            q = (
                select(Ranking, TrainingRun.completed_at)
                .join(TrainingRun, TrainingRun.id == Ranking.run_id)
                .where(TrainingRun.status == "completed")
                .order_by(
                    Ranking.instance, Ranking.metric, Ranking.horizon,
                    TrainingRun.completed_at.desc(),
                )
            )
            rows = list(s.execute(q))

        grouped: dict[tuple[str, str, str], list[Any]] = {}
        for r in rows:
            key = (r.Ranking.instance, r.Ranking.metric, r.Ranking.horizon)
            grouped.setdefault(key, []).append(r)

        summary: list[dict[str, Any]] = []
        for (inst, met, hor), recents in grouped.items():
            current = recents[0]
            winner = current.Ranking.winning_algo
            previous = None
            winner_since = current.completed_at
            for r in recents[1:]:
                if r.Ranking.winning_algo == winner:
                    winner_since = r.completed_at
                    continue
                previous = r.Ranking.winning_algo
                break
            window = recents[: max(1, recent_window)]
            unique_recent = len({r.Ranking.winning_algo for r in window})
            # Top-3 from current ranking, for quick "near miss" inspection
            top3 = (current.Ranking.ranked or [])[:3]
            summary.append({
                "instance": inst,
                "metric": met,
                "horizon": hor,
                "current_winner": winner,
                "previous_winner": previous,
                "winner_since": winner_since.isoformat() if winner_since else None,
                "unique_winners_recent": unique_recent,
                "recent_window_runs": len(window),
                "current_top3": top3,
            })
        summary.sort(key=lambda r: (r["instance"], r["metric"], r["horizon"]))
        return summary

    # ----- dashboard aggregates -----

    def system_overview(self) -> dict[str, Any]:
        """One-shot stats for the dashboard Overview cards.

        Cheap aggregate queries; safe to call on every scrape/refresh.
        """
        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(days=1)
        with self.session() as s:
            total = s.scalar(select(func.count(TrainingRun.id))) or 0
            runs_24h = s.scalar(
                select(func.count(TrainingRun.id))
                .where(TrainingRun.started_at >= day_ago)
                .where(TrainingRun.status == "completed")
            ) or 0
            failed_24h = s.scalar(
                select(func.count(TrainingRun.id))
                .where(TrainingRun.started_at >= day_ago)
                .where(TrainingRun.status == "failed")
            ) or 0
            running = s.scalar(
                select(func.count(TrainingRun.id))
                .where(TrainingRun.status.in_(("pending", "running")))
            ) or 0
            uniq_instances = s.scalar(
                select(func.count(func.distinct(TrainingRun.instance)))
                .where(TrainingRun.status == "completed")
            ) or 0
            uniq_metrics = s.scalar(
                select(func.count(func.distinct(TrainingRun.metric)))
                .where(TrainingRun.status == "completed")
            ) or 0
            uniq_horizons = s.scalar(
                select(func.count(func.distinct(TrainingRun.horizon)))
                .where(TrainingRun.status == "completed")
            ) or 0
            # Target count = distinct (instance, metric, horizon) across completed runs
            target_rows = s.execute(
                select(TrainingRun.instance, TrainingRun.metric, TrainingRun.horizon)
                .where(TrainingRun.status == "completed")
                .distinct()
            ).all()
            unique_targets = len(target_rows)
            # Avg run duration over last 50 completed runs
            recent = list(s.scalars(
                select(TrainingRun)
                .where(TrainingRun.status == "completed")
                .where(TrainingRun.duration_seconds.is_not(None))
                .order_by(TrainingRun.completed_at.desc())
                .limit(50)
            ))
            avg_dur = (
                sum(r.duration_seconds for r in recent) / len(recent)
                if recent else None
            )
        return {
            "runs_total": int(total),
            "runs_24h": int(runs_24h),
            "failed_24h": int(failed_24h),
            "running_now": int(running),
            "unique_instances": int(uniq_instances),
            "unique_metrics": int(uniq_metrics),
            "unique_horizons": int(uniq_horizons),
            "unique_targets": unique_targets,
            "avg_run_duration": avg_dur,
            "duration_sample_size": len(recent),
        }

    def model_stats(
        self,
        *,
        metric: str | None = None,
        horizon: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Per-algorithm aggregate within an optional (metric, horizon, time) filter."""
        with self.session() as s:
            # Constrain the source runs first; downstream tables filter
            # through `run_id IN (subq)`.
            run_q = select(TrainingRun.id).where(TrainingRun.status == "completed")
            if metric:
                run_q = run_q.where(TrainingRun.metric == metric)
            if horizon:
                run_q = run_q.where(TrainingRun.horizon == horizon)
            if since is not None:
                run_q = run_q.where(TrainingRun.completed_at >= since)
            if until is not None:
                run_q = run_q.where(TrainingRun.completed_at <= until)
            run_ids = list(s.scalars(run_q))

            if not run_ids:
                return []

            metric_rows = list(s.scalars(
                select(RunMetric)
                .where(RunMetric.fold == -1)
                .where(RunMetric.run_id.in_(run_ids))
            ))
            art_rows = list(s.scalars(
                select(ModelArtifact).where(ModelArtifact.run_id.in_(run_ids))
            ))
            ranking_rows = list(s.scalars(
                select(Ranking).where(Ranking.run_id.in_(run_ids))
            ))

        runs_per_algo: dict[str, set[int]] = {}
        score_sum: dict[str, dict[str, float]] = {}
        score_cnt: dict[str, dict[str, int]] = {}
        for m in metric_rows:
            runs_per_algo.setdefault(m.algo, set()).add(m.run_id)
            score_sum.setdefault(m.algo, {}).setdefault(m.score_metric, 0.0)
            score_cnt.setdefault(m.algo, {}).setdefault(m.score_metric, 0)
            if m.value is None:
                continue
            score_sum[m.algo][m.score_metric] += float(m.value)
            score_cnt[m.algo][m.score_metric] += 1

        dur_sum: dict[str, float] = {}
        dur_cnt: dict[str, int] = {}
        for a in art_rows:
            dur_sum[a.algo] = dur_sum.get(a.algo, 0.0) + a.train_duration_seconds
            dur_cnt[a.algo] = dur_cnt.get(a.algo, 0) + 1

        wins: dict[str, int] = {}
        for r in ranking_rows:
            wins[r.winning_algo] = wins.get(r.winning_algo, 0) + 1

        algos = sorted(set(runs_per_algo) | set(wins) | set(dur_sum))
        out: list[dict[str, Any]] = []
        for a in algos:
            n_runs = len(runs_per_algo.get(a, set()))
            n_wins = wins.get(a, 0)
            out.append({
                "algo": a,
                "wins": n_wins,
                "runs": n_runs,
                "win_rate": (n_wins / n_runs) if n_runs else 0.0,
                "avg_mae": (score_sum.get(a, {}).get("mae", 0.0) / score_cnt[a]["mae"])
                            if score_cnt.get(a, {}).get("mae") else None,
                "avg_rmse": (score_sum.get(a, {}).get("rmse", 0.0) / score_cnt[a]["rmse"])
                            if score_cnt.get(a, {}).get("rmse") else None,
                "avg_train_duration": (dur_sum[a] / dur_cnt[a]) if dur_cnt.get(a) else None,
            })
        return out

    def wins_by_metric(
        self,
        *,
        metric: str | None = None,
        horizon: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, dict[str, int]]:
        """Returns {metric: {algo: win_count}}. All filters optional."""
        with self.session() as s:
            q = select(Ranking).join(TrainingRun, TrainingRun.id == Ranking.run_id)
            if metric:
                q = q.where(Ranking.metric == metric)
            if horizon:
                q = q.where(Ranking.horizon == horizon)
            if since:
                q = q.where(TrainingRun.completed_at >= since)
            if until:
                q = q.where(TrainingRun.completed_at <= until)
            rows = list(s.scalars(q))
        out: dict[str, dict[str, int]] = {}
        for r in rows:
            bucket = out.setdefault(r.metric, {})
            bucket[r.winning_algo] = bucket.get(r.winning_algo, 0) + 1
        return out

    def run_per_fold_scores(self, run_id: int) -> dict[str, list[dict[str, Any]]]:
        """Per-fold backtest scores for every algo in this run.

        Output: {algo: [{fold, mae, rmse, mape, smape, r2}, ...]}.
        Folds are zero-indexed (oldest expanding-window fold = 0).
        """
        with self.session() as s:
            rows = list(s.scalars(
                select(RunMetric)
                .where(RunMetric.run_id == run_id)
                .where(RunMetric.fold >= 0)
            ))
        nested: dict[str, dict[int, dict[str, Any]]] = {}
        for m in rows:
            folds = nested.setdefault(m.algo, {})
            cell = folds.setdefault(m.fold, {"fold": m.fold})
            cell[m.score_metric] = m.value
        return {
            algo: [folds[i] for i in sorted(folds)]
            for algo, folds in nested.items()
        }

    def run_full_detail(self, run_id: int) -> dict[str, Any] | None:
        """All data needed to render the Run Detail page."""
        with self.session() as s:
            run = s.get(TrainingRun, run_id)
            if run is None:
                return None
            metrics = list(s.scalars(
                select(RunMetric).where(RunMetric.run_id == run_id).where(RunMetric.fold == -1)
            ))
            artifacts = list(s.scalars(
                select(ModelArtifact).where(ModelArtifact.run_id == run_id)
            ))
            ranking = s.scalar(select(Ranking).where(Ranking.run_id == run_id))

        scores_by_algo: dict[str, dict[str, float | None]] = {}
        for m in metrics:
            scores_by_algo.setdefault(m.algo, {})[m.score_metric] = m.value
        composite_by_algo: dict[str, float | None] = {}
        if ranking and ranking.ranked:
            for r in ranking.ranked:
                composite_by_algo[r["algo"]] = r.get("composite")
        duration_by_algo: dict[str, float] = {
            a.algo: a.train_duration_seconds for a in artifacts
        }
        artifact_by_algo: dict[str, str] = {a.algo: a.path for a in artifacts}
        algos = sorted(
            set(scores_by_algo) | set(composite_by_algo) | set(duration_by_algo),
            key=lambda a: (composite_by_algo.get(a) is None,
                           -(composite_by_algo.get(a) or 0)),
        )
        winner = ranking.winning_algo if ranking else None
        return {
            "run": {
                "id": run.id,
                "instance": run.instance,
                "metric": run.metric,
                "horizon": run.horizon,
                "status": run.status,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "duration_seconds": run.duration_seconds,
                "error": run.error,
                "config_snapshot": run.config_snapshot or {},
            },
            "rows": [
                {
                    "algo": a,
                    "is_winner": (a == winner),
                    "composite": composite_by_algo.get(a),
                    "scores": scores_by_algo.get(a, {}),
                    "duration": duration_by_algo.get(a),
                    "artifact": artifact_by_algo.get(a),
                }
                for a in algos
            ],
        }

    def instance_summary(self, recent_window: int = 10) -> list[dict[str, Any]]:
        """One row per instance with per-(metric, horizon) status grid.

        Output shape per instance:
            {
              "instance": "fake-1",
              "targets":  N,                # distinct (metric, horizon) seen
              "completed": N, "failed": N, "running": N,  # recent run statuses
              "last_run_at": <iso str>,
              "winners":     {"cpu/medium": "lstm", ...},
              "stability":   {"cpu/medium": 1, ...},   # unique_winners_recent
            }
        """
        summary_rows = self.winners_summary(recent_window=recent_window)
        per_inst: dict[str, dict[str, Any]] = {}
        for row in summary_rows:
            inst = row["instance"]
            bucket = per_inst.setdefault(inst, {
                "instance": inst,
                "winners": {},
                "stability": {},
                "last_run_at": None,
                "targets": 0,
            })
            key = f"{row['metric']}/{row['horizon']}"
            bucket["winners"][key] = row.get("current_winner")
            bucket["stability"][key] = row.get("unique_winners_recent", 0)
            bucket["targets"] += 1
            ts = row.get("winner_since")
            if ts and (bucket["last_run_at"] is None or ts > bucket["last_run_at"]):
                bucket["last_run_at"] = ts

        # Recent run status counts per instance (last 50 runs / instance)
        with self.session() as s:
            recent_runs = list(s.scalars(
                select(TrainingRun)
                .order_by(TrainingRun.started_at.desc())
                .limit(2000)
            ))
        for r in recent_runs:
            bucket = per_inst.setdefault(r.instance, {
                "instance": r.instance,
                "winners": {},
                "stability": {},
                "last_run_at": None,
                "targets": 0,
            })
            bucket.setdefault("completed", 0)
            bucket.setdefault("failed", 0)
            bucket.setdefault("running", 0)
            if r.status == "completed":
                bucket["completed"] += 1
            elif r.status == "failed":
                bucket["failed"] += 1
            elif r.status in ("pending", "running"):
                bucket["running"] += 1

        out = sorted(per_inst.values(), key=lambda x: x["instance"])
        return out

    def instance_detail(self, instance: str, recent_window: int = 10, run_limit: int = 50) -> dict[str, Any] | None:
        """Everything to render /ui/instances/{instance}."""
        full = self.winners_summary(recent_window=recent_window)
        rows = [r for r in full if r["instance"] == instance]
        if not rows:
            # Maybe the instance exists in runs but never produced a ranking
            with self.session() as s:
                any_run = s.scalar(
                    select(TrainingRun).where(TrainingRun.instance == instance).limit(1)
                )
            if not any_run:
                return None
        with self.session() as s:
            recent = list(s.scalars(
                select(TrainingRun)
                .where(TrainingRun.instance == instance)
                .order_by(TrainingRun.started_at.desc())
                .limit(run_limit)
            ))
        return {
            "instance": instance,
            "targets": rows,
            "recent_runs": [
                {
                    "id": r.id, "metric": r.metric, "horizon": r.horizon,
                    "status": r.status,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                    "duration_seconds": r.duration_seconds,
                    "error": r.error,
                }
                for r in recent
            ],
        }

    def runs_filtered(
        self,
        *,
        instance: str | None = None,
        metric: str | None = None,
        horizon: str | None = None,
        status: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        sort: str = "started_at",
        direction: str = "desc",
        limit: int = 200,
    ) -> list[TrainingRun]:
        """Like list_runs but with date range + sortable columns."""
        with self.session() as s:
            q = select(TrainingRun)
            if instance:
                q = q.where(TrainingRun.instance.ilike(f"%{instance}%"))
            if metric:
                q = q.where(TrainingRun.metric == metric)
            if horizon:
                q = q.where(TrainingRun.horizon == horizon)
            if status:
                q = q.where(TrainingRun.status == status)
            if since:
                q = q.where(TrainingRun.started_at >= since)
            if until:
                q = q.where(TrainingRun.started_at <= until)
            col = {
                "id": TrainingRun.id,
                "instance": TrainingRun.instance,
                "metric": TrainingRun.metric,
                "horizon": TrainingRun.horizon,
                "status": TrainingRun.status,
                "started_at": TrainingRun.started_at,
                "completed_at": TrainingRun.completed_at,
                "duration_seconds": TrainingRun.duration_seconds,
            }.get(sort, TrainingRun.started_at)
            q = q.order_by(col.desc() if direction == "desc" else col.asc()).limit(limit)
            return list(s.scalars(q))

    def error_groups(self, hours: int = 24, limit: int = 200) -> list[dict[str, Any]]:
        """Group failed-run errors by their first-line signature for triage."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self.session() as s:
            rows = list(s.scalars(
                select(TrainingRun)
                .where(TrainingRun.status == "failed")
                .where(TrainingRun.started_at >= cutoff)
                .order_by(TrainingRun.started_at.desc())
                .limit(limit)
            ))
        groups: dict[str, dict[str, Any]] = {}
        for r in rows:
            sig = (r.error or "unknown").split("\n", 1)[0][:160]
            g = groups.setdefault(sig, {
                "signature": sig, "count": 0,
                "first_seen": None, "last_seen": None,
                "sample_instances": [],
            })
            g["count"] += 1
            ts = r.started_at.isoformat() if r.started_at else None
            if ts:
                g["last_seen"] = max(g["last_seen"] or "", ts)
                g["first_seen"] = min(g["first_seen"] or "ZZZ", ts)
            if r.instance not in g["sample_instances"] and len(g["sample_instances"]) < 5:
                g["sample_instances"].append(r.instance)
        return sorted(groups.values(), key=lambda x: -x["count"])

    def attention_targets(self, recent_window: int = 10, stale_after_hours: int = 24) -> list[dict[str, Any]]:
        """Targets that warrant a human look: flapping, recently failed, or stale."""
        now = datetime.now(timezone.utc)
        stale_cutoff = now - timedelta(hours=stale_after_hours)
        out: list[dict[str, Any]] = []
        for row in self.winners_summary(recent_window=recent_window):
            last_iso = row.get("winner_since")
            try:
                last_dt = datetime.fromisoformat(last_iso) if last_iso else None
            except (TypeError, ValueError):
                last_dt = None
            # SQLite strips tzinfo on round-trip; treat naive timestamps as UTC.
            if last_dt and last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            reasons: list[str] = []
            if row["unique_winners_recent"] >= 3:
                reasons.append("flapping")
            if last_dt and last_dt < stale_cutoff:
                reasons.append("stale")
            if reasons:
                out.append({**row,
                            "reason": reasons[0],
                            "last_completed_at": last_iso})
        # Also surface targets whose latest run failed.
        with self.session() as s:
            recent_failed = list(s.scalars(
                select(TrainingRun)
                .where(TrainingRun.status == "failed")
                .where(TrainingRun.started_at >= now - timedelta(hours=24))
                .order_by(TrainingRun.started_at.desc())
                .limit(50)
            ))
        seen = {(r["instance"], r["metric"], r["horizon"]) for r in out}
        for r in recent_failed:
            key = (r.instance, r.metric, r.horizon)
            if key in seen:
                continue
            out.append({
                "instance": r.instance,
                "metric": r.metric,
                "horizon": r.horizon,
                "current_winner": None,
                "unique_winners_recent": 0,
                "last_completed_at": r.started_at.isoformat() if r.started_at else None,
                "reason": "failed",
            })
            seen.add(key)
        return out

    # ----- settings overrides (UI-managed config) -----

    def get_all_settings_overrides(self) -> dict[str, Any]:
        """Return the entire overrides table as a flat {dotted_key: value} dict."""
        with self.session() as s:
            rows = list(s.scalars(select(SettingsOverride)))
        return {r.key: r.value for r in rows}

    def set_settings_override(self, key: str, value: Any, updated_by: str | None = None) -> None:
        with self.session() as s:
            existing = s.get(SettingsOverride, key)
            if existing:
                existing.value = value
                existing.updated_at = datetime.now(timezone.utc)
                existing.updated_by = updated_by
            else:
                s.add(SettingsOverride(key=key, value=value, updated_by=updated_by))

    def delete_settings_override(self, key: str) -> None:
        with self.session() as s:
            row = s.get(SettingsOverride, key)
            if row:
                s.delete(row)

    # ----- target overrides (per-target enable + cron) -----

    def get_target_overrides(self) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = list(s.scalars(select(TargetOverride)))
        return [
            {
                "instance": r.instance, "metric": r.metric, "horizon": r.horizon,
                "enabled": r.enabled, "schedule_cron": r.schedule_cron,
                "note": r.note,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                "updated_by": r.updated_by,
            }
            for r in rows
        ]

    def get_target_overrides_map(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        """Returns {(instance, metric, horizon): {enabled, schedule_cron, ...}}."""
        return {
            (r["instance"], r["metric"], r["horizon"]): r
            for r in self.get_target_overrides()
        }

    def upsert_target_override(
        self,
        *,
        instance: str,
        metric: str,
        horizon: str,
        enabled: bool | None = None,
        schedule_cron: str | None = None,
        note: str | None = None,
        updated_by: str | None = None,
    ) -> None:
        with self.session() as s:
            row = s.get(TargetOverride, (instance, metric, horizon))
            if row is None:
                row = TargetOverride(
                    instance=instance, metric=metric, horizon=horizon,
                    enabled=True if enabled is None else enabled,
                    schedule_cron=schedule_cron,
                    note=note,
                    updated_by=updated_by,
                )
                s.add(row)
            else:
                if enabled is not None:
                    row.enabled = enabled
                row.schedule_cron = schedule_cron
                row.note = note
                row.updated_at = datetime.now(timezone.utc)
                row.updated_by = updated_by

    def delete_target_override(self, instance: str, metric: str, horizon: str) -> None:
        with self.session() as s:
            row = s.get(TargetOverride, (instance, metric, horizon))
            if row:
                s.delete(row)

    def is_target_enabled(self, instance: str, metric: str, horizon: str) -> bool:
        """Default-on: a target without an override is enabled."""
        with self.session() as s:
            row = s.get(TargetOverride, (instance, metric, horizon))
        return True if row is None else bool(row.enabled)
