"""账号：邀请码注册、密码登录，会话与学习记忆按账号隔离。

为什么要有账号：共享口令只能回答"能不能进来"，回答不了"这是谁"。
认不出人，就没法追问题、没法单独踢人、换台设备历史会话就没了。

模式由 ``LA_AUTH`` 决定：

- ``accounts``：用户名 + 密码，注册需要邀请码。
- ``code``：沿用共享口令 ``LA_ACCESS_TOKEN``，只适合临时演示。
- ``off``：不鉴权，只在本机开发用。
- ``auto``（默认）：库里已有账号或还有可用邀请码就走 accounts，
  否则有口令走 code，都没有就 off。

密码用标准库 ``hashlib.scrypt`` 加盐存哈希（不引入依赖）；登录态存库，
服务重启不掉线，cookie 里只放随机令牌，库里存它的 sha256。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import re
import secrets
import sqlite3
import time
from datetime import UTC, datetime
from urllib.parse import quote
from uuid import uuid4

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE = "la_session"
SESSION_DAYS = 30
MIN_TOKEN = 20
MIN_PASSWORD = 8
MAX_PASSWORD = 200
MAX_FAILURES = 8
FAILURE_WINDOW = 300
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1

# 用户名：2–32 字符，字母/数字/汉字开头，可含 _ . -
USERNAME = re.compile(r"^[0-9A-Za-z\u4e00-\u9fff][0-9A-Za-z\u4e00-\u9fff_.\-]{1,31}$")

OPEN_PATHS = {"/login", "/register", "/logout", "/healthz", "/api/me", "/survey", "/api/survey"}
OPEN_PREFIXES = ("/static/",)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    password TEXT NOT NULL,
    created_at TEXT NOT NULL,
    invite_code TEXT,
    last_login TEXT,
    disabled INTEGER NOT NULL DEFAULT 0);
CREATE UNIQUE INDEX IF NOT EXISTS users_name ON users(username COLLATE NOCASE);
CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    user_agent TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS auth_sessions_user ON auth_sessions(user_id);
CREATE TABLE IF NOT EXISTS invite_codes (
    code TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    max_uses INTEGER NOT NULL DEFAULT 1,
    uses INTEGER NOT NULL DEFAULT 0,
    disabled INTEGER NOT NULL DEFAULT 0);
"""


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def check_password(stored: str, password: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        target = base64.b64decode(digest)
        attempt = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
            dklen=len(target),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(attempt, target)


_DUMMY: str | None = None


def _burn_time(password: str) -> None:
    """用户名不存在时也付同样的哈希开销，别让响应时间泄漏账号是否存在。"""
    global _DUMMY
    if _DUMMY is None:
        _DUMMY = hash_password("not-a-real-password")
    check_password(_DUMMY, password)


def username_error(username: str) -> str:
    if not USERNAME.match(username):
        return "用户名需 2–32 位，字母、数字或汉字开头，可含 _ . -"
    return ""


def password_error(password: str) -> str:
    if len(password) < MIN_PASSWORD:
        return f"密码至少 {MIN_PASSWORD} 位"
    if len(password) > MAX_PASSWORD:
        return f"密码最多 {MAX_PASSWORD} 位"
    return ""


class AccountStore:
    """账号、登录态、邀请码。挂在 ``Store`` 上，和其他 store 共用同一个 sqlite。"""

    def init_accounts(self):
        with self.connect() as db:
            db.executescript(SCHEMA)

    def has_users(self) -> bool:
        with self.connect() as db:
            return db.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None

    def signup_open(self) -> bool:
        with self.connect() as db:
            row = db.execute(
                "SELECT 1 FROM invite_codes WHERE disabled=0 AND uses<max_uses LIMIT 1"
            ).fetchone()
        return row is not None

    def users(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT u.id,u.username,u.created_at,u.last_login,u.disabled,u.invite_code,"
                "(SELECT COUNT(*) FROM conversations c WHERE c.owner='u:'||u.id) AS conversations "
                "FROM users u ORDER BY u.created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def user_by_name(self, username: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM users WHERE username=? COLLATE NOCASE", (username.strip(),)
            ).fetchone()
        return dict(row) if row else None

    def user_by_id(self, user_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None

    def create_user(self, username: str, password: str, *, invite: str = "") -> dict:
        name = username.strip()
        problem = username_error(name) or password_error(password)
        if problem:
            raise ValueError(problem)
        secret = hash_password(password)
        user = {
            "id": str(uuid4()),
            "username": name,
            "created_at": datetime.now(UTC).isoformat(),
            "invite_code": invite.strip() or None,
        }
        with self.connect() as db:
            try:
                db.execute(
                    "INSERT INTO users(id,username,password,created_at,invite_code) VALUES (?,?,?,?,?)",
                    (user["id"], name, secret, user["created_at"], user["invite_code"]),
                )
            except sqlite3.IntegrityError:
                raise ValueError("用户名已被占用") from None
        return user

    def register(self, username: str, password: str, invite: str) -> dict:
        """邀请码注册：核销邀请码和建号在同一个事务里，不会出现码用掉人没建成。"""
        name = username.strip()
        problem = username_error(name) or password_error(password)
        if problem:
            raise ValueError(problem)
        code = invite.strip()
        if not code:
            raise ValueError("请填写邀请码")
        secret = hash_password(password)
        user = {
            "id": str(uuid4()),
            "username": name,
            "created_at": datetime.now(UTC).isoformat(),
            "invite_code": code,
        }
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM invite_codes WHERE code=?", (code,)).fetchone()
            if not row or row["disabled"] or row["uses"] >= row["max_uses"]:
                raise ValueError("邀请码无效或已用完")
            if db.execute("SELECT 1 FROM users WHERE username=? COLLATE NOCASE", (name,)).fetchone():
                raise ValueError("用户名已被占用")
            db.execute("UPDATE invite_codes SET uses=uses+1 WHERE code=?", (code,))
            db.execute(
                "INSERT INTO users(id,username,password,created_at,invite_code) VALUES (?,?,?,?,?)",
                (user["id"], name, secret, user["created_at"], code),
            )
        return user

    def set_password(self, user_id: str, password: str) -> None:
        problem = password_error(password)
        if problem:
            raise ValueError(problem)
        with self.connect() as db:
            db.execute(
                "UPDATE users SET password=? WHERE id=?", (hash_password(password), user_id)
            )
            # 改密码等于把旧设备全部踢下线
            db.execute("DELETE FROM auth_sessions WHERE user_id=?", (user_id,))

    def set_disabled(self, user_id: str, disabled: bool) -> None:
        with self.connect() as db:
            db.execute("UPDATE users SET disabled=? WHERE id=?", (1 if disabled else 0, user_id))
            if disabled:
                db.execute("DELETE FROM auth_sessions WHERE user_id=?", (user_id,))

    def start_session(self, user_id: str, *, days: int = SESSION_DAYS, agent: str = "") -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self.connect() as db:
            db.execute("DELETE FROM auth_sessions WHERE expires_at<?", (now,))
            db.execute(
                "INSERT INTO auth_sessions(token_hash,user_id,created_at,expires_at,user_agent) "
                "VALUES (?,?,?,?,?)",
                (_digest(token), user_id, now, now + days * 86400, agent[:200]),
            )
            db.execute(
                "UPDATE users SET last_login=? WHERE id=?",
                (datetime.now(UTC).isoformat(), user_id),
            )
        return token

    def session_user(self, token: str) -> dict | None:
        if len(token) < MIN_TOKEN:
            return None
        digest = _digest(token)
        with self.connect() as db:
            row = db.execute(
                "SELECT u.id,u.username,u.disabled,s.expires_at FROM auth_sessions s "
                "JOIN users u ON u.id=s.user_id WHERE s.token_hash=?",
                (digest,),
            ).fetchone()
            if not row:
                return None
            if row["expires_at"] < time.time() or row["disabled"]:
                db.execute("DELETE FROM auth_sessions WHERE token_hash=?", (digest,))
                return None
        return {"id": row["id"], "username": row["username"]}

    def end_session(self, token: str) -> None:
        if len(token) < MIN_TOKEN:
            return
        with self.connect() as db:
            db.execute("DELETE FROM auth_sessions WHERE token_hash=?", (_digest(token),))

    def end_user_sessions(self, user_id: str) -> int:
        with self.connect() as db:
            cursor = db.execute("DELETE FROM auth_sessions WHERE user_id=?", (user_id,))
        return cursor.rowcount or 0

    def create_invite(self, *, note: str = "", max_uses: int = 1) -> str:
        code = secrets.token_urlsafe(9)
        with self.connect() as db:
            db.execute(
                "INSERT INTO invite_codes(code,created_at,note,max_uses) VALUES (?,?,?,?)",
                (code, datetime.now(UTC).isoformat(), note[:100], max(1, int(max_uses))),
            )
        return code

    def invites(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM invite_codes ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def disable_invite(self, code: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE invite_codes SET disabled=1 WHERE code=?", (code.strip(),))


def resolve(settings, store) -> str:
    mode = settings.auth
    if mode != "auto":
        return mode
    if store.has_users() or store.signup_open():
        return "accounts"
    if settings.access_token.get_secret_value().strip():
        return "code"
    return "off"


def current(request: Request) -> dict | None:
    return getattr(request.state, "user", None)


def owner_key(request: Request) -> str:
    """会话与记忆的归属键：登录用户是 ``u:<id>``，匿名模式退回访客 cookie。"""
    user = current(request)
    if user:
        return "u:" + user["id"]
    from learning_agent import visitor

    return visitor.current(request)


STYLE = """
:root{font:15px/1.7 system-ui;color:#20312d;background:#f5f6f2}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px}
main{background:white;border:1px solid #dce3dc;border-radius:12px;padding:28px;width:100%;max-width:420px}
h1{font-size:19px;margin:0 0 6px}p{margin:8px 0;color:#53675e}
label{display:block;margin-top:14px;font-weight:600}
input{font:inherit;width:100%;padding:10px;border:1px solid #cbd5cc;border-radius:6px;margin-top:6px;box-sizing:border-box}
button{font:inherit;width:100%;margin-top:20px;padding:10px;border:1px solid #cbd5cc;border-radius:6px;background:#e7efe7;cursor:pointer}
.error{color:#9b3d29}small{color:#64756b;display:block;margin-top:16px}
a{color:#38634b}
"""


def _layout(title: str, body: str) -> str:
    return (
        '<!doctype html>\n<html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Learning Agent · {html.escape(title)}</title>"
        f'<link rel="icon" href="data:,"><style>{STYLE}</style>'
        f"<main>{body}</main></html>"
    )


def _safe_target(value: str) -> str:
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    return value


def login_page(error: str = "", target: str = "/") -> str:
    block = f'<p class="error">{html.escape(error)}</p>' if error else ""
    link = html.escape(target, quote=True)
    return _layout(
        "登录",
        f"<h1>Learning Agent</h1><p>用自己的账号登录；资料共用，会话与学习记录各自可见。</p>"
        f"{block}<form method=\"post\" action=\"/login\">"
        f'<input type="hidden" name="next" value="{link}">'
        "<label for=\"username\">用户名</label>"
        '<input id="username" name="username" autocomplete="username" required autofocus>'
        "<label for=\"password\">密码</label>"
        '<input id="password" name="password" type="password" autocomplete="current-password" required>'
        "<button type=\"submit\">登录</button></form>"
        f'<small>还没有账号？<a href="/register?next={link}">用邀请码注册</a></small>',
    )


def register_page(error: str = "", target: str = "/", *, open_for_signup: bool = True) -> str:
    link = html.escape(target, quote=True)
    if not open_for_signup:
        return _layout(
            "注册",
            "<h1>注册暂未开放</h1><p>当前没有可用的邀请码。请联系把网址发给你的人要一个。</p>"
            f'<small><a href="/login?next={link}">返回登录</a></small>',
        )
    block = f'<p class="error">{html.escape(error)}</p>' if error else ""
    return _layout(
        "注册",
        f"<h1>注册</h1><p>需要邀请码。注册后会话、学习主题与记忆只有你自己能看到。</p>{block}"
        "<form method=\"post\" action=\"/register\">"
        f'<input type="hidden" name="next" value="{link}">'
        '<label for="invite">邀请码</label>'
        '<input id="invite" name="invite" autocomplete="off" required autofocus>'
        '<label for="username">用户名</label>'
        '<input id="username" name="username" autocomplete="username" required>'
        f'<label for="password">密码（至少 {MIN_PASSWORD} 位）</label>'
        '<input id="password" name="password" type="password" autocomplete="new-password" required>'
        '<label for="confirm">再输一次密码</label>'
        '<input id="confirm" name="confirm" type="password" autocomplete="new-password" required>'
        '<button type="submit">注册并登录</button></form>'
        f'<small>已有账号？<a href="/login?next={link}">直接登录</a></small>',
    )


def install(app, settings, store) -> str:
    """按 ``LA_AUTH`` 装鉴权，返回实际生效的模式。"""
    mode = resolve(settings, store)

    @app.get("/api/me")
    async def me(request: Request):
        user = current(request)
        return {
            "mode": mode,
            "authenticated": bool(user),
            "id": user["id"] if user else None,
            "username": user["username"] if user else None,
            "signup": mode == "accounts" and store.signup_open(),
        }

    if mode != "accounts":
        return mode

    failures: dict[str, list[float]] = {}

    def blocked(key: str) -> bool:
        now = time.time()
        recent = [moment for moment in failures.get(key, []) if now - moment < FAILURE_WINDOW]
        failures[key] = recent
        return len(recent) >= MAX_FAILURES

    def failed(key: str) -> None:
        failures.setdefault(key, []).append(time.time())

    def forget(*keys: str) -> None:
        for key in keys:
            failures.pop(key, None)

    def hand_off(response, user: dict, request: Request):
        token = store.start_session(
            user["id"], days=settings.session_days, agent=request.headers.get("user-agent", "")
        )
        response.set_cookie(
            COOKIE, token, max_age=settings.session_days * 86400, httponly=True, samesite="lax"
        )
        return response

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        token = request.cookies.get(COOKIE, "")
        request.state.user = store.session_user(token)
        if request.state.user or path in OPEN_PATHS or path.startswith(OPEN_PREFIXES):
            return await call_next(request)
        if path.startswith("/api/") or "application/json" in request.headers.get("accept", ""):
            return JSONResponse({"detail": "请先登录"}, status_code=401)
        target = path + (f"?{request.url.query}" if request.url.query else "")
        return RedirectResponse(f"/login?next={quote(target)}", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(next: str = ""):
        return login_page(target=_safe_target(next))

    @app.get("/register", response_class=HTMLResponse)
    async def register_form(next: str = ""):
        return register_page(target=_safe_target(next), open_for_signup=store.signup_open())

    @app.post("/login", response_class=HTMLResponse)
    async def login_submit(request: Request):
        form = await request.form()
        target = _safe_target(str(form.get("next") or "/"))
        name = str(form.get("username") or "").strip()[:64]
        password = str(form.get("password") or "")[:MAX_PASSWORD]
        host = request.client.host if request.client else "-"
        keys = (f"ip:{host}", f"user:{name.lower()}")
        if blocked(keys[0]) or blocked(keys[1]):
            return HTMLResponse(login_page("尝试次数过多，请五分钟后再试", target), 429)
        user = store.user_by_name(name)
        if user:
            passed = check_password(user["password"], password) and not user["disabled"]
        else:
            _burn_time(password)
            passed = False
        if not passed:
            failed(keys[0])
            failed(keys[1])
            return HTMLResponse(login_page("用户名或密码不正确", target), 401)
        forget(*keys)
        return hand_off(RedirectResponse(target, 303), user, request)

    @app.post("/register", response_class=HTMLResponse)
    async def register_submit(request: Request):
        form = await request.form()
        target = _safe_target(str(form.get("next") or "/"))
        name = str(form.get("username") or "").strip()[:64]
        password = str(form.get("password") or "")[:MAX_PASSWORD]
        confirm = str(form.get("confirm") or "")[:MAX_PASSWORD]
        invite = str(form.get("invite") or "").strip()[:100]
        host = request.client.host if request.client else "-"
        if not store.signup_open():
            return HTMLResponse(register_page(target=target, open_for_signup=False), 403)
        if blocked(f"signup:{host}"):
            return HTMLResponse(register_page("尝试次数过多，请一小时后再试", target), 429)
        if password != confirm:
            failed(f"signup:{host}")
            return HTMLResponse(register_page("两次输入的密码不一致", target), 422)
        try:
            user = store.register(name, password, invite)
        except ValueError as error:
            failed(f"signup:{host}")
            return HTMLResponse(register_page(str(error), target), 422)
        return hand_off(RedirectResponse(target, 303), user, request)

    @app.post("/logout")
    async def logout(request: Request):
        store.end_session(request.cookies.get(COOKIE, ""))
        response = RedirectResponse("/login", 303)
        response.delete_cookie(COOKIE)
        return response

    return mode
