#!/usr/bin/env python3
"""账号运维：发邀请码、建号、重置密码、停用账号。

同学自己用邀请码注册，这里只处理例外情况。在项目根目录跑：

  python scripts/accounts.py list
  python scripts/accounts.py invite --note 3班 --uses 10
  python scripts/accounts.py invites
  python scripts/accounts.py add alice
  python scripts/accounts.py passwd alice
  python scripts/accounts.py disable alice

默认读 ``LA_DATA_ROOT``（与 start.sh 一致）；指到别处用 ``--data-root``。
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learning_agent.config import Settings
from learning_agent.store import Store


def store_for(args) -> Store:
    root = Path(args.data_root) if args.data_root else Settings(_env_file=None).data_root
    if not (root / "workspace.db").exists():
        raise SystemExit(f"数据目录里没有 workspace.db：{root.resolve()}")
    return Store(root)


def new_password() -> str:
    return secrets.token_urlsafe(9)


def cmd_list(store, args):
    users = store.users()
    if not users:
        print("还没有账号。用 invite 发码，或 add 直接建号。")
    for user in users:
        state = "已停用" if user["disabled"] else "正常"
        print(
            f"{user['username']:<20} {state:<4} 会话 {user['conversations']:<4} "
            f"注册 {user['created_at'][:10]}  最近登录 {(user['last_login'] or '-')[:16]}  "
            f"邀请码 {user['invite_code'] or '-'}"
        )


def cmd_invite(store, args):
    code = store.create_invite(note=args.note, max_uses=args.uses)
    print(f"邀请码：{code}（可用 {max(1, args.uses)} 次，备注：{args.note or '无'}）")
    print("发给同学时提醒：注册后资料仍共用，但会话与学习记录只有自己可见。")


def cmd_invites(store, args):
    codes = store.invites()
    if not codes:
        print("还没有邀请码。")
    for code in codes:
        state = "已作废" if code["disabled"] else (
            "已用完" if code["uses"] >= code["max_uses"] else "可用"
        )
        print(
            f"{code['code']:<14} {state:<4} {code['uses']}/{code['max_uses']} 次  "
            f"{code['note'] or '-'}  {code['created_at'][:10]}"
        )


def cmd_revoke(store, args):
    store.disable_invite(args.code)
    print(f"已作废邀请码 {args.code}")


def cmd_add(store, args):
    password = args.password or new_password()
    store.create_user(args.username, password, invite="(cli)")
    print(f"已建号 {args.username}")
    print(f"初始密码：{password}")
    print("把这个密码单独发过去，让对方登录后自己改。")


def cmd_passwd(store, args):
    user = store.user_by_name(args.username)
    if not user:
        raise SystemExit(f"没有这个账号：{args.username}")
    password = args.password or new_password()
    store.set_password(user["id"], password)
    print(f"{user['username']} 的新密码：{password}")
    print("旧登录态已全部失效。")


def cmd_disable(store, args):
    user = store.user_by_name(args.username)
    if not user:
        raise SystemExit(f"没有这个账号：{args.username}")
    store.set_disabled(user["id"], True)
    print(f"已停用 {user['username']}，其登录态已失效；数据保留。")


def cmd_enable(store, args):
    user = store.user_by_name(args.username)
    if not user:
        raise SystemExit(f"没有这个账号：{args.username}")
    store.set_disabled(user["id"], False)
    print(f"已恢复 {user['username']}。")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Learning Agent 账号运维")
    parser.add_argument("--data-root", help="数据目录，默认取 LA_DATA_ROOT 或 ./data")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="列出账号").set_defaults(run=cmd_list)

    invite = commands.add_parser("invite", help="生成邀请码（注册用）")
    invite.add_argument("--note", default="", help="备注，比如班级或用途")
    invite.add_argument("--uses", type=int, default=1, help="可用次数，默认 1")
    invite.set_defaults(run=cmd_invite)

    commands.add_parser("invites", help="列出邀请码").set_defaults(run=cmd_invites)

    revoke = commands.add_parser("revoke", help="作废邀请码")
    revoke.add_argument("code")
    revoke.set_defaults(run=cmd_revoke)

    add = commands.add_parser("add", help="直接建号（不走邀请码）")
    add.add_argument("username")
    add.add_argument("--password", help="指定密码；不给就随机生成")
    add.set_defaults(run=cmd_add)

    passwd = commands.add_parser("passwd", help="重置密码，踢掉旧登录态")
    passwd.add_argument("username")
    passwd.add_argument("--password", help="指定密码；不给就随机生成")
    passwd.set_defaults(run=cmd_passwd)

    for name, run, text in (
        ("disable", cmd_disable, "停用账号"),
        ("enable", cmd_enable, "恢复账号"),
    ):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("username")
        sub.set_defaults(run=run)

    args = parser.parse_args(argv)
    args.run(store_for(args), args)


if __name__ == "__main__":
    main()
