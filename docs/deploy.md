# 部署到云服务器

给自己的电脑用，`./start.sh` 就够了。这篇讲的是另一种情况：把服务放到一台一直在线的机器上，用链接给别人试用。

不用 GPU。模型调用走 HTTP，机器只负责跑 FastAPI、SQLite，以及把 PDF 页渲染成图片。

## 需要什么

- 一台 2 核 2 G 的 Linux 服务器（Ubuntu 24.04 / Debian 12），系统盘 20 G 起。2 G 是下限，`uv sync` 解包时内存会冲高
- 一个 OpenAI 兼容的模型服务：`base URL`、模型名、密钥
- 一个访问口令（自己生成，别用生日）

在阿里云上买「轻量应用服务器」最省事：选 Ubuntu 24.04，2 核 2 G，价格约 30–60 元/月，学生认证后更便宜。轻量自带的防火墙页面就是安全组，放行规则在那一页加。

## 一、本地打包

```bash
./deploy/pack.sh
```

产物在 `dist/learning-agent-trial.tar.gz`，约 31 M，包含代码、依赖锁和演示工作区。

只打演示工作区，不打 `data/evaluations/`——那里有 2.6 G 真实课程 corpus，是跑评测用的，放服务器上除了占盘没别的用。想换数据用 `LA_DEMO_DATA=/path/to/workspace ./deploy/pack.sh`。

## 二、上传解压

```bash
scp dist/learning-agent-trial.tar.gz root@<公网IP>:/srv/
ssh root@<公网IP> 'cd /srv && tar xzf learning-agent-trial.tar.gz'
```

解压后是 `/srv/learning-agent-trial/`，`data/workspace.db` 应该在里面。

## 三、写配置

```bash
ssh root@<公网IP>
cd /srv/learning-agent-trial
cp .env.example .env
chmod 600 .env
vi .env
```

要改的四个值：

```ini
LA_BASE_URL=https://surplustoken.com/v1
LA_MODEL=gpt-6-sol
LA_API_KEY=你的密钥
LA_ACCESS_TOKEN=这里填访问口令
LA_HOST=0.0.0.0
```

口令自己生成一个：

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(9))'
```

这两条别漏：

- `LA_ACCESS_TOKEN` 留空就等于没设防。`install.sh` 会因此拒绝继续，不是它多事——公开 IP 上加个空口令，等于把全部会话和资料挂在网上。
- `LA_HOST` 要是 `0.0.0.0`。留在 `127.0.0.1` 时外面连不进来，表现是「端口开了但一直转圈」。

试用阶段不需要向量检索，`.env` 里保持 `LA_RETRIEVAL_MODE=bm25` 和 `LA_EMBEDDING_PROVIDER=disabled` 就行。开 `hybrid` 得再部署一个 embedding 服务，而试用阶段检索质量不是瓶颈。

## 四、起服务

```bash
./deploy/install.sh
```

它会装 ffmpeg 和 uv、同步依赖、装一个 systemd 服务并启动，最后做两项自检：

```
GET /        -> 303（跳登录页，对）
GET /api/spaces -> 401（未登录挡住，对）
```

看到 `303` 和 `401` 才说明口令真的在起作用。两个都返回 `200` 就是没生效，先回头查 `LA_ACCESS_TOKEN`。

常用命令：

```bash
systemctl status learning-agent
journalctl -u learning-agent -f
systemctl restart learning-agent
```

### 用 Docker 的话

`Dockerfile` 和 `docker-compose.yml` 也在包里，二选一：

```bash
docker compose up -d --build
docker compose ps
```

第一次构建要拉镜像装依赖，五到十分钟。`api` 和 `worker` 都得是 `running`——`worker` 负责 PDF 转图和音视频转码，它没起来的话上传的 PDF 会一直停在「准备中」。轻量服务器上 Docker 会多占几百兆内存，2 G 的机器我更建议走 `install.sh`。

## 五、放行端口

在云控制台的防火墙 / 安全组里放行 **TCP 8320**。轻量应用服务器在实例详情的「防火墙」页，ECS 在安全组里，两处都只开这一个端口。

然后访问 `http://<公网IP>:8320`，输入 `LA_ACCESS_TOKEN` 里的口令。

用 IP 加端口不需要备案；绑域名走 80/443 才需要。试用阶段没必要买域名。

## 六、更新

代码改了之后重新打包上传：

```bash
./deploy/pack.sh
scp dist/learning-agent-trial.tar.gz root@<公网IP>:/srv/
ssh root@<公网IP> 'cd /srv && tar xzf learning-agent-trial.tar.gz && systemctl restart learning-agent'
```

`.env` 不在包里，覆盖解压不会把它冲掉。但 `data/` 会被包里的版本覆盖——要保留服务器上积累的会话和问卷，先备份（下一节）。

## 七、备份

`data/` 不在 git 里，删了就没了（这个项目真丢过一次）。每天打一个包，留最近七份：

```bash
cat >/etc/cron.daily/learning-agent-backup <<'EOF'
#!/bin/sh
set -eu
cd /srv/learning-agent-trial
mkdir -p backups
tar czf "backups/data-$(date +%F).tar.gz" data
ls -1t backups/data-*.tar.gz | tail -n +8 | xargs -r rm -f
EOF
chmod +x /etc/cron.daily/learning-agent-backup
```

恢复：

```bash
tar xzf backups/data-2026-11-18.tar.gz -C /srv/learning-agent-trial
```

问卷结果单独导一份，方便直接读：

```bash
uv run python -m learning_agent.survey --workspace data --output survey.json
```

## 几个会被问到的问题

**为什么不用本地模型？** 每轮最多把三张 PDF 页面图发给模型。8B 级别的模型在 2 G 内存的机器上跑不动，租 GPU 比买 token 贵一个数量级，讲图质量还更差。

**一个口令给所有人，会不会互相看到？** 会。所有访客共用一个资料空间，会话列表是共享的。登录页写明了「不要上传私人文件」。真要做隔离得给空间加归属，那是另一件事。

**重启会踢人吗？** 会。会话只存在进程内存里，重启后所有人都要重新输口令。这是有意为之：省掉会话存储，也避免旧 cookie 长期有效。

**同时能撑几个人？** 瓶颈不在 CPU，在模型 API 的等待时间。一轮几十秒，几个人同时提问不会把机器压满。
