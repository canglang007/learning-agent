"""共享口令：``LA_AUTH=code`` 时启用，所有人共用一个口令。

账号模式（``accounts.py``）是默认推荐；这里留着是为了"临时挡一下无关访问"这种场景。
没有 ``LA_ACCESS_TOKEN`` 时 ``install`` 直接返回，不注册登录页——不假装受保护。
"""

from __future__ import annotations

import hmac
import html
import secrets
import time

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE = "la_access"
OPEN_PATHS = {"/login", "/logout", "/healthz"}
OPEN_PREFIXES = ("/static/",)
SESSION_SECONDS = 7 * 24 * 3600
MAX_FAILURES = 8
FAILURE_WINDOW = 300

PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Learning Agent · 输入访问口令</title>
<link rel="icon" href="data:,">
<style>
:root{{font:15px/1.7 system-ui;color:#20312d;background:#f5f6f2}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px}}
form{{background:white;border:1px solid #dce3dc;border-radius:12px;padding:28px;width:100%;max-width:420px}}
h1{{font-size:19px;margin:0 0 6px}}p{{margin:8px 0;color:#53675e}}
input{{font:inherit;width:100%;padding:10px;border:1px solid #cbd5cc;border-radius:6px;margin:10px 0}}
button{{font:inherit;width:100%;padding:10px;border:1px solid #cbd5cc;border-radius:6px;background:#e7efe7;cursor:pointer}}
.error{{color:#9b3d29}}small{{color:#64756b;display:block;margin-top:14px}}
</style>
<form method="post" action="/login">
<h1>Learning Agent</h1>
<p>这是一个对内测开放的演示环境，需要访问口令。</p>
{error}
<input type="hidden" name="next" value="{next}">
<label for="code">访问口令</label>
<input id="code" name="code" type="password" autocomplete="current-password" required autofocus>
<button type="submit">进入</button>
<small>资料空间是大家共用的，请不要上传私人文件或真实作业；你的会话只有你自己能看到。</small>
</form>
</html>
"""


def _page(error: str, target: str) -> str:
    block = f'<p class="error">{html.escape(error)}</p>' if error else ""
    return PAGE.format(error=block, next=html.escape(target, quote=True))


def _safe_target(value: str) -> str:
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    return value


def install(app, settings):
    code = settings.access_token.get_secret_value().strip()
    if not code:
        return

    sessions: dict[str, float] = {}
    failures: dict[str, list[float]] = {}

    def authorized(request: Request) -> bool:
        token = request.cookies.get(COOKIE)
        if not token:
            return False
        deadline = sessions.get(token)
        if deadline is None:
            return False
        if deadline < time.time():
            sessions.pop(token, None)
            return False
        return True

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if path in OPEN_PATHS or path.startswith(OPEN_PREFIXES) or authorized(request):
            return await call_next(request)
        if path.startswith("/api/") or "application/json" in request.headers.get("accept", ""):
            return JSONResponse({"detail": "需要访问口令，请先打开页面输入口令"}, status_code=401)
        target = path + (f"?{request.url.query}" if request.url.query else "")
        return RedirectResponse(f"/login?next={target}", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(next: str = ""):
        return _page("", _safe_target(next))

    @app.post("/login", response_class=HTMLResponse)
    async def login_submit(request: Request):
        form = await request.form()
        target = _safe_target(str(form.get("next") or "/"))
        candidate = str(form.get("code") or "")[:200]
        host = request.client.host if request.client else "-"
        now = time.time()
        recent = [moment for moment in failures.get(host, []) if now - moment < FAILURE_WINDOW]
        if len(recent) >= MAX_FAILURES:
            return HTMLResponse(_page("尝试次数过多，请五分钟后再试", target), status_code=429)
        if not hmac.compare_digest(candidate.encode(), code.encode()):
            recent.append(now)
            failures[host] = recent
            return HTMLResponse(_page("口令不正确", target), status_code=401)
        failures.pop(host, None)
        token = secrets.token_urlsafe(32)
        sessions[token] = now + SESSION_SECONDS
        response = RedirectResponse(target, status_code=303)
        response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True, samesite="lax")
        return response

    @app.post("/logout")
    async def logout(request: Request):
        token = request.cookies.get(COOKIE)
        if token:
            sessions.pop(token, None)
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE)
        return response
