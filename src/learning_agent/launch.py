"""Local first-run configuration and owned API/worker process supervision."""

import argparse
import getpass
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

from learning_agent.config import Settings


def doctor(settings, port=8320):
    with socket.socket() as sock:
        occupied = sock.connect_ex(("127.0.0.1", port)) == 0
    return {
        "python": sys.version.split()[0],
        "configured": bool(settings.api_key.get_secret_value()),
        "model": settings.model,
        "protocol": settings.protocol,
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
        "uvicorn": importlib.util.find_spec("uvicorn") is not None,
        "port": port,
        "port_occupied": occupied,
        "data_root": str(settings.data_root.resolve()),
        "network_model_calls": 0,
    }


def local_address():
    """Best-effort LAN address; no packet is sent for a UDP connect."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return ""


def write_config(path, values):
    path = Path(path)
    lines = path.read_text().splitlines() if path.exists() else []
    for key, value in values.items():
        if "\n" in value or "\r" in value:
            raise ValueError("配置不能含换行")
        encoded = key + "=" + json.dumps(value, ensure_ascii=False)
        positions = [i for i, line in enumerate(lines) if line.startswith(key + "=")]
        if positions:
            lines[positions[0]] = encoded
            for i in reversed(positions[1:]):
                lines.pop(i)
        else:
            lines.append(encoded)
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)


def configure(path=Path(".env")):
    current = Settings(_env_file=path)
    print("首次配置：填写OpenAI兼容服务。密钥隐藏输入，仅保存于本机.env。")
    url = input(f"API base URL [{current.base_url}]：").strip() or current.base_url
    parts = urlsplit(url)
    if (
        parts.scheme not in {"https", "http"}
        or not parts.netloc
        or parts.username
        or parts.password
        or parts.query
    ):
        raise ValueError("base URL需要http(s)服务地址，不在网址中填写密钥")
    model = input(f"模型ID [{current.model}]：").strip() or current.model
    protocol = (
        input(f"协议 chat_completions / responses [{current.protocol}]：").strip()
        or current.protocol
    )
    if protocol not in {"chat_completions", "responses"}:
        raise ValueError("未知协议")
    key = getpass.getpass("API key（已配置时留空保留）：") or current.api_key.get_secret_value()
    if not key:
        raise ValueError("尚未填写API key")
    # Preserve unrelated ASR/retrieval/context settings. Configuration does not probe paid APIs.
    write_config(
        path, {"LA_BASE_URL": url, "LA_MODEL": model, "LA_PROTOCOL": protocol, "LA_API_KEY": key}
    )
    print("配置已保存，尚未发送模型请求。可在网页首次提问时验证该服务。")


def stop_children(children):
    for child in children:
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + 10
    for child in children:
        if child.poll() is None:
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    parser = argparse.ArgumentParser(description="Learning Agent：一次启动API和媒体worker")
    parser.add_argument("--port", type=int, default=8320)
    parser.add_argument(
        "--host",
        default=None,
        help="监听地址；默认127.0.0.1，对外试用用0.0.0.0（请同时设置访问口令）",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--configure", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口需要在1到65535之间")
    try:
        if args.configure:
            configure()
        settings = Settings()
        if args.data_root:
            settings = settings.model_copy(update={"data_root": args.data_root.resolve()})
        host = args.host or settings.host
        info = doctor(settings, args.port)
        info["host"] = host
        # 实际生效的鉴权方式：auto 要看库里有没有账号或邀请码。
        from learning_agent.accounts import resolve
        from learning_agent.store import Store

        auth_mode = resolve(settings, Store(settings.data_root))
        info["auth"] = auth_mode
        if args.doctor:
            print(json.dumps(info, ensure_ascii=False, indent=2))
            return
        if not info["configured"]:
            if sys.stdin.isatty():
                configure()
                settings = Settings().model_copy(update={"data_root": settings.data_root})
            else:
                # 没有密钥也应能打开界面看演示数据和历史会话；提问会由模型层给出明确失败，
                # 而不是让第一次克隆仓库的人卡在启动这一步。
                print(
                    "未配置模型服务：界面、演示资料和历史会话仍可打开，提问会明确失败。"
                    "配置方法：./start.sh --configure（密钥只写入本机 .env）。",
                    flush=True,
                )
        if info["port_occupied"]:
            parser.error(
                f"端口{args.port}已被占用，可查看 http://127.0.0.1:{args.port} 或使用--port选择另一个端口；未结束已有服务"
            )
        if not info["uvicorn"]:
            parser.error("缺少uvicorn，请在项目目录运行uv sync")
        if not info["ffmpeg"] or not info["ffprobe"]:
            print("未检测到FFmpeg：文字/PDF可使用；音视频处理需安装ffmpeg/ffprobe。")
        env = {**os.environ, "LA_DATA_ROOT": str(settings.data_root.resolve())}
        if args.evaluation_root:
            env["LA_EVALUATION_ROOT"] = str(args.evaluation_root.resolve())
        children = []

        def stopped(*_):
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, stopped)
        try:
            children.append(
                subprocess.Popen(
                    [sys.executable, "-m", "learning_agent.worker"], env=env, start_new_session=True
                )
            )
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "learning_agent.app:create_app",
                        "--factory",
                        "--host",
                        host,
                        "--port",
                        str(args.port),
                    ],
                    env=env,
                    start_new_session=True,
                )
            )
            url = f"http://127.0.0.1:{args.port}"
            for _ in range(100):
                if any(c.poll() is not None for c in children):
                    raise RuntimeError("API或worker启动失败，已结束本次启动的进程")
                with socket.socket() as sock:
                    if sock.connect_ex(("127.0.0.1", args.port)) == 0:
                        break
                time.sleep(0.1)
            else:
                raise RuntimeError("API启动超时")
            print(
                f"已启动：{url}；评测查看：{url}/evaluation。Ctrl-C同时结束API与worker。",
                flush=True,
            )
            if host not in {"127.0.0.1", "localhost"}:
                lan = local_address()
                if lan:
                    print(f"局域网地址：http://{lan}:{args.port}", flush=True)
                if auth_mode == "off":
                    print(
                        "注意：已监听非本机地址但没有任何鉴权（LA_AUTH=off 且没设口令），"
                        "同网段任何人都能读写全部会话。",
                        flush=True,
                    )
                elif auth_mode == "code":
                    print(
                        "当前用共享口令模式：所有人共用一个口令，认不出谁是谁；"
                        "要按人隔离请改用 LA_AUTH=accounts。",
                        flush=True,
                    )
            if not args.no_browser:
                webbrowser.open(url)
            while all(c.poll() is None for c in children):
                time.sleep(0.25)
            raise RuntimeError("API或worker已退出，正在结束本次启动的其他进程")
        except KeyboardInterrupt:
            pass
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            stop_children(children)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
