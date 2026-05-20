from forecaster.config.schema import RankingConfig
from forecaster.evaluation.ranking import rank_models


def make_cfg() -> RankingConfig:
    return RankingConfig(
        metrics=["mae", "rmse", "mape", "smape", "r2"],
        weights={"mae": 0.25, "rmse": 0.25, "mape": 0.2, "smape": 0.2, "r2": 0.1},
        direction={"mae": "min", "rmse": "min", "mape": "min", "smape": "min", "r2": "max"},
    )


def test_perfect_winner_first():
    scored = {
        "good": {"mae": 0.1, "rmse": 0.1, "mape": 1.0, "smape": 1.0, "r2": 0.99},
        "bad":  {"mae": 9.9, "rmse": 9.9, "mape": 99.0, "smape": 99.0, "r2": 0.0},
    }
    ranked = rank_models(scored, make_cfg())
    assert ranked[0].algo == "good"
    assert ranked[1].algo == "bad"
    assert ranked[0].rank == 1


def test_handles_single_candidate():
    scored = {"only": {"mae": 1.0, "rmse": 1.0, "mape": 5.0, "smape": 5.0, "r2": 0.5}}
    ranked = rank_models(scored, make_cfg())
    assert len(ranked) == 1
    assert ranked[0].algo == "only"


def test_skips_missing_metric_gracefully():
    scored = {
        "partial": {"mae": 1.0, "rmse": 1.0},
        "full":    {"mae": 1.0, "rmse": 1.0, "mape": 5.0, "smape": 5.0, "r2": 0.9},
    }
    ranked = rank_models(scored, make_cfg())
    assert {r.algo for r in ranked} == {"partial", "full"}
