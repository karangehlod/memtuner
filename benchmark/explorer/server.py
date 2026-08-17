"""Explorer server — local web server for benchmark visualization.

Serves a read-only dashboard for browsing benchmark results.
Depends on the optional `explorer` extra (fastapi + uvicorn).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from benchmark.explorer.data_loader import ExplorerDataLoader


def create_explorer_app(results_directory: Path) -> Any:
    """Create the FastAPI application for the explorer.

    Args:
        results_directory: Path to the results directory.

    Returns:
        A FastAPI application instance.

    Raises:
        ImportError: If fastapi is not installed.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as import_error:
        raise ImportError(
            "Explorer requires the 'explorer' optional dependency. "
            "Install with: pip install -e '.[explorer]'"
        ) from import_error

    data_loader = ExplorerDataLoader(results_directory)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Load data on startup, yield control, no teardown needed."""
        data_loader.load_all_runs()
        yield

    app = FastAPI(
        title="Agentic Memory Benchmark Explorer",
        description="Read-only dashboard for benchmark results",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        runs = data_loader.list_run_ids()
        run_list = "".join(f'<li><a href="/api/runs/{rid}">{rid}</a></li>' for rid in runs)
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>Benchmark Explorer</title>
        <style>
            body {{ font-family: system-ui; max-width: 800px; margin: 2rem auto; }}
            h1 {{ color: #2563eb; }}
            .metric {{ display: inline-block; padding: 1rem; margin: 0.5rem;
                       background: #f3f4f6; border-radius: 0.5rem; }}
            .metric .value {{ font-size: 2rem; font-weight: bold; color: #1f2937; }}
            .metric .label {{ font-size: 0.875rem; color: #6b7280; }}
        </style>
        </head>
        <body>
            <h1>🧠 Benchmark Explorer</h1>
            <h2>Loaded Runs ({len(runs)})</h2>
            <ul>{run_list}</ul>
            <h2>API Endpoints</h2>
            <ul>
                <li><code>GET /api/runs</code> — List all runs</li>
                <li><code>GET /api/runs/{{run_id}}</code> — Get run details</li>
                <li><code>GET /api/metrics/{{metric_name}}</code> — Metric series</li>
            </ul>
        </body>
        </html>
        """

    @app.get("/api/runs")
    async def list_runs() -> JSONResponse:
        runs = data_loader.list_run_ids()
        return JSONResponse(content={"runs": runs, "count": len(runs)})

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        result = data_loader.get_run(run_id)
        if result is None:
            return JSONResponse(
                content={"error": f"Run not found: {run_id}"},
                status_code=404,
            )
        return JSONResponse(content=result.model_dump(mode="json"))

    @app.get("/api/metrics/{metric_name}")
    async def get_metric_series(metric_name: str) -> JSONResponse:
        series = data_loader.get_metric_series(metric_name)
        return JSONResponse(content={"metric": metric_name, "series": series})

    @app.post("/api/reload")
    async def reload_data() -> JSONResponse:
        runs = data_loader.load_all_runs()
        return JSONResponse(content={"reloaded": True, "count": len(runs)})

    return app


def run_explorer_server(
    results_directory: Path,
    host: str = "127.0.0.1",
    port: int = 8501,
) -> None:
    """Start the explorer server.

    Args:
        results_directory: Path to the results directory.
        host: Host to bind to.
        port: Port to listen on.

    Raises:
        ImportError: If uvicorn is not installed.
    """
    try:
        import uvicorn
    except ImportError as import_error:
        raise ImportError(
            "Explorer requires uvicorn. Install with: pip install -e '.[explorer]'"
        ) from import_error

    app = create_explorer_app(results_directory)
    uvicorn.run(app, host=host, port=port)
