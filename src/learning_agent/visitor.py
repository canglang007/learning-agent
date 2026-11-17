"""给每个浏览器发一个稳定标识，匿名模式下用它来决定"这条会话是谁建的"。

有账号时归属用 ``u:<user_id>``（见 ``accounts.owner_key``），这个 cookie 只作兜底：
``LA_AUTH=off``（本地开发）或共享口令模式下，没有账号可认，只能按浏览器分。
它本身不参与鉴权——口令由 ``access.py``、账号由 ``accounts.py`` 负责。

标识存在 cookie 里，服务重启不丢。老数据 ``owner=''`` 视为公共（示例会话、迁移前的记录），所有人可见。
"""

from __future__ import annotations

import secrets

from fastapi import Request

COOKIE = "la_visitor"
LIFETIME = 180 * 24 * 3600
MIN_LENGTH = 16


def install(app):
    @app.middleware("http")
    async def identify(request: Request, call_next):
        token = request.cookies.get(COOKIE) or ""
        fresh = len(token) < MIN_LENGTH
        if fresh:
            token = secrets.token_urlsafe(18)
        request.state.visitor = token
        response = await call_next(request)
        if fresh:
            response.set_cookie(
                COOKIE, token, max_age=LIFETIME, httponly=True, samesite="lax"
            )
        return response


def current(request: Request) -> str:
    return getattr(request.state, "visitor", "")
