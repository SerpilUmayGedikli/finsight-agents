from typing import cast

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ExceptionHandler

from apps.api.errors import (
    APIError,
    api_error_handler,
    http_error_handler,
    validation_error_handler,
)
from apps.api.logging_config import configure_logging
from apps.api.middleware import RequestIDMiddleware
from apps.api.routes import ask_router, ingest_router, search_router
from apps.api.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title="Finsight API", version="0.1.0")
    app.add_middleware(RequestIDMiddleware)
    app.add_exception_handler(APIError, cast(ExceptionHandler, api_error_handler))
    app.add_exception_handler(
        StarletteHTTPException, cast(ExceptionHandler, http_error_handler)
    )
    app.add_exception_handler(
        RequestValidationError, cast(ExceptionHandler, validation_error_handler)
    )
    app.state.ingest_jobs = {}
    app.include_router(search_router)
    app.include_router(ask_router)
    app.include_router(ingest_router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
