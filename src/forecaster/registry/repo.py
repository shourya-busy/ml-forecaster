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

from .models import Base, Forecast, ModelArtifact, Ranking, RunMetric, TrainingRun


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
    ) -> list[Forecast]:
        """Return forecasts whose run is the latest completed for each
        (instance, metric, horizon). Used by API + Prom exposition.
        """
        with self.session() as s:
            # Latest completed run id per target.
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

    def model_stats(self) -> list[dict[str, Any]]:
        """Per-algorithm aggregate: total wins, runs entered, avg MAE/RMSE, avg train time."""
        with self.session() as s:
            # All averaged-fold metric rows
            metric_rows = list(s.scalars(
                select(RunMetric).where(RunMetric.fold == -1)
            ))
            # Artifact rows give us train durations
            art_rows = list(s.scalars(select(ModelArtifact)))
            # Winning algo per (completed) ranking row
            ranking_rows = list(s.scalars(select(Ranking)))

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

    def wins_by_metric(self) -> dict[str, dict[str, int]]:
        """Returns {metric: {algo: win_count, ...}, ...} for stacked bar chart."""
        with self.session() as s:
            rows = list(s.scalars(select(Ranking)))
        out: dict[str, dict[str, int]] = {}
        for r in rows:
            bucket = out.setdefault(r.metric, {})
            bucket[r.winning_algo] = bucket.get(r.winning_algo, 0) + 1
        return out

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
