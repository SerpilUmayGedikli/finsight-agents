import logging
import time
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_logger = logging.getLogger("apps.api.request")


def get_request_id() -> str:
    return _request_id.get()


class RequestIDMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        request_id = headers.get(b"x-request-id", b"").decode("latin-1").strip()
        request_id = request_id or str(uuid.uuid4())
        token = _request_id.set(request_id)
        start = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"x-request-id"
                ]
                response_headers.append((b"x-request-id", request_id.encode("latin-1")))
                message["headers"] = response_headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            _logger.exception("request_failed")
            raise
        finally:
            _logger.info(
                "http_request",
                extra={
                    "method": scope["method"],
                    "path": scope.get("path", ""),
                    "status_code": status_code,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )
            _request_id.reset(token)
