"""FastAPI app factory + uvicorn entrypoint."""

from __future__ import annotations

import os
import signal

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from ..config.loader import reload_settings
from ..observability.logging import configure_logging
from ..ui import router as ui_router, static_app as ui_static_app
from .routes import config as cfg_route
from .routes import diagnostics as diagnostics_route
from .routes import forecasts as forecasts_route
from .routes import health as health_route
from .routes import models as models_route
from .routes import rankings as rankings_route
from .routes import runs as runs_route
from .prometheus_export import router as metrics_router


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="forecaster",
        version="0.1.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.include_router(health_route.router)
    app.include_router(metrics_router)
    app.include_router(runs_route.router)
    app.include_router(rankings_route.router)
    app.include_router(forecasts_route.router)
    app.include_router(models_route.router)
    app.include_router(cfg_route.router)
    app.include_router(diagnostics_route.router)
    app.include_router(ui_router)
    app.mount("/ui/static", ui_static_app, name="ui-static")

    @app.get("/", include_in_schema=False)
    def _root() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    def _sighup(_s, _f):
        reload_settings()
    signal.signal(signal.SIGHUP, _sighup)

    return app


# Module-level app for `uvicorn forecaster.api.main:app`
app = create_app()


def run() -> None:
    import uvicorn
    host = os.environ.get("API_HOST", "0.0.0.0")  # noqa: S104 - container expects 0.0.0.0
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("forecaster.api.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    run()
