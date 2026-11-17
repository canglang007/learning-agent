# 部署到云服务器

给自己的电脑用，`./start.sh` 就够了。这篇讲的是另一种情况：把服务放到一台一直在线的机器上，用链接给别人试用。

不用 GPU。模型调用走 HTTP，机器只负责跑 FastAPI、SQLite 和把 PDF 页渲染成图片。

## 需要什么

- 一台 2 核 4G 的 Linux 服务器（Ubuntu 24.04 或 Debian 12），系统盘 40 G 起
- Docker 与 Compose 插件
- 一个 OpenAI 兼容的模型服务：`base URL`、模型名、密钥
- 一个访问口令（自己生成，不要用生日）

内存不能低于 2 G：`uv sync` 期间会同时解包依赖，1 G 的机器容易直接 OOM。CPU 差一点只是慢。

## 一、准备数据

只传试用要用到的那一份，不要把整个 `data/` 搬上去——里面 `evaluations/` 有 2.6 G，是跑评测用的，和试用无关。

本地打包：

```bash
cd /path/to/learning-agent
tar czf trial-data.tar.gz -C data/demo/workspace .
```

传上去并解开，作为容器里的 `/app/data`：

```bash
ssh root@<你的服务器> 'mkdir -p /srv/learning-agent/data'
scp trial-data.tar.gz root@<你的服务器>:/srv/learning-agent/
ssh root@<你的服务器> 'tar xzf /srv/learning-agent/trial-data.tar.gz -C /srv/learning-agent/data'
```

解开后 `/srv/learning-agent/data/workspace.db` 应该存在，`materials/` 里是素材文件。

## 二、准备代码

```bash
ssh root@<你的服务器>
cd /srv/learning-agent
git clone https://github.com/<你>/learning-agent.git .
```

私有仓库要先在服务器上配好 deploy key 或 token。传压缩包也行，但以后更新就得重传。

## 三、写配置

在服务器上 `/srv/learning-agent/.env`：

```ini
LA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LA_MODEL=qwen4-max
LA_PROTOCOL=chat_completions
LA_API_KEY=你的密钥

LA_ACCESS_TOKEN=这里填访问口令
LA_RETRIEVAL_MODE=bm25
LA_EMBEDDING_PROVIDER=disabled
```

```bash
chmod 600 .env
```

生成口令：

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(9))'
```

`LA_ACCESS_TOKEN` 留空就等于不设防，`docker-compose.yml` 里也有同名变量，两个地方别只设一个。配了之后所有人先看到登录页，输入口令才能进。

试用阶段把向量检索关掉（`bm25` + `disabled`）。开 `hybrid` 就要再部署一个 embedding 服务，机器和时间都翻倍，而试用阶段检索质量不是瓶颈。

## 四、起服务

```bash
cd /srv/learning-agent
docker compose up -d --build
docker compose ps
```

第一次构建要拉 Python 镜像和装依赖，五到十分钟。起来之后：

```bash
curl -s localhost:8320/healthz
```

返回 `{"ok":true}` 就正常。`api` 和 `worker` 都应该是 `running`——`worker` 负责 PDF 转图、音视频转码，它没起来的话上传的 PDF 会一直停在“准备中”。

## 五、放行端口

在云控制台的**安全组**里放行 TCP 8320（不是防火墙里加，是安全组，两层都要）。

然后用 `http://<公网IP>:8320` 访问。第一次打开会跳登录页，输入 `LA_ACCESS_TOKEN` 里的口令。

用 IP 加端口访问不需要备案；绑定域名走 80/443 才需要。试用阶段没必要买域名。

## 六、更新与回滚

```bash
git pull
docker compose up -d --build
```

数据在宿主机目录里，容器重建不会丢。但**重建不会备份**——下面这件事必须做。

## 七、备份

`data/` 不在 git 里，删了就没了（这个项目真丢过一次）。每天打一个包，保留最近七份：

```bash
cat >/etc/cron.daily/learning-agent-backup <<'EOF'
#!/bin/sh
set -eu
cd /srv/learning-agent
mkdir -p backups
tar czf "backups/data-$(date +%F).tar.gz" data
ls -1t backups/data-*.tar.gz | tail -n +8 | xargs -r rm -f
EOF
chmod +x /etc/cron.daily/learning-agent-backup
```

恢复就是把包解回 `data/`：

```bash
tar xzf backups/data-2026-11-18.tar.gz -C /srv/learning-agent
```

## 几个会被问到的问题

**为什么不用本地模型？** 这个项目每轮最多把三张 PDF 页面图发给模型。8B 级别的本地模型在 4 G 内存的机器上跑不动，租 GPU 比直接买 token 贵一个数量级，讲图质量还更差。

**一个口令给所有人，会不会互相看到？** 会。所有访客共用一个资料空间，会话列表是共享的。所以登录页写明了“不要上传私人文件”。要做真正隔离得给空间加归属，那是另一件事。

**重启会踢人吗？** 会。会话只存在进程内存里，容器重启后所有人都要重新输入口令。这是有意为之，省掉会话存储，也避免旧 cookie 长期有效。

**同时能撑几个人？** 瓶颈不在 CPU 而在模型 API 的并发和等待时间。几个人同时提问没问题，一轮要几十秒，不会把机器压满。
