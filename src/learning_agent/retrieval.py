"""Persistent source-aligned chunks, BM25 and dense rank fusion for small local courses."""

import asyncio
import json
import math
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx

from learning_agent.config import Settings
from learning_agent.jobs import Jobs, configuration
from learning_agent.routing import course_codes

INDEX_VERSION = 2


def tokenize(text):
    words = re.findall(
        r"[a-z0-9_]+|[\u3400-\u9fff]+", unicodedata.normalize("NFKC", text).casefold()
    )
    tokens = []
    for word in words:
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            tokens.extend(
                word if len(word) == 1 else [word[i : i + 2] for i in range(len(word) - 1)]
            )
        else:
            tokens.append(word)
    return tokens


class Bm25Index:
    """语料统计与查询打分分开：统计一次建好，之后每次查询只算分数。

    原先每次 ``bm25(query, texts)`` 都把全部块重新分词。单讲几十块时无所谓，
    整讲语料扩到 900 多块、又叠上子查询与精排对照之后，一次提问会白烧一两秒
    的纯重复计算。统计只与语料有关，和查询无关，所以由调用方按语料版本缓存。

    参数与原先一致（k1=1.2、b=0.75），分数口径没有变。
    """

    def __init__(self, texts, k1=1.2, b=0.75):
        self.documents = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(document.values()) for document in self.documents]
        self.total = len(self.documents)
        self.average = sum(self.lengths) / max(1, self.total) or 1
        self.frequencies = Counter(word for document in self.documents for word in document)
        self.k1 = k1
        self.b = b

    def scores(self, query):
        scores = []
        for document, length in zip(self.documents, self.lengths):
            score = 0.0
            for word in set(tokenize(query)):
                frequency = document[word]
                if frequency:
                    idf = math.log(
                        1
                        + (self.total - self.frequencies[word] + 0.5)
                        / (self.frequencies[word] + 0.5)
                    )
                    score += (
                        idf
                        * frequency
                        * (self.k1 + 1)
                        / (frequency + self.k1 * (1 - self.b + self.b * length / self.average))
                    )
            scores.append(score)
        return scores


def bm25(query, texts):
    return Bm25Index(texts).scores(query)


def fuse(*rankings, k=60):
    scores = Counter()
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] += 1 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i)), scores


def course_scope(query, chunks):
    """Match explicit course identifiers to filenames, not to the lecture cover text."""
    identifiers = course_codes(c["name"] for c in chunks) & set(tokenize(query))
    if not identifiers:
        return chunks, query, []
    selected = [
        c for c in chunks if identifiers.intersection(re.split(r"[^a-z0-9]+", c["name"].casefold()))
    ]
    cleaned = query
    for identifier in identifiers:
        cleaned = re.sub(
            r"(?<![a-z0-9])" + re.escape(identifier) + r"(?![a-z0-9])",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
    return selected, cleaned.strip() or query, sorted({c["id"] for c in selected})


class EmbeddingError(Exception):
    pass


class RerankError(Exception):
    pass


def sigmoid(value):
    import math

    return 1 / (1 + math.exp(-value))


def combined_scores(ce_scores, pool, weight):
    """CE 概率与归一化 RRF 位次的分数空间融合，返回池内每个位置的分数。

    对外展示的 ``score`` 必须用它与排序键一致——否则会出现"分数大的排在后面"：
    只返回 CE 概率时，一个 CE 略低但位次更靠前的候选会排在前面。
    ``weight=1.0`` 时退化为 CE 概率本身，``weight=0`` 时为纯 RRF 位次分。
    """
    rrf = {j: 1 / (60 + j + 1) for j in range(len(pool))}
    rrf_max = max(rrf.values()) if rrf else 1
    return {
        j: weight * sigmoid(ce_scores[j]) + (1 - weight) * rrf[j] / rrf_max
        for j in range(len(pool))
    }


def fuse_ce_with_rrf(ce_scores, pool, weight):
    """分数空间融合 CE 概率与 RRF 位次分数，返回池内新顺序（位置序列）。

    纯 CE 替换会把"转述型查询"（词面与目标重叠少）的目标挤出前 5——
    17 题对照里 3 题如此；位次空间等权融合又救不回它们。
    分数空间融合在 CE 高置信时跟随 CE、置信度接近时由 RRF 托底：
    权重 0.85–0.9 时 14 题可评分用例 hit@5 全中（纯替换 12/14）。
    ``weight=1.0`` 退化为纯 CE 替换，``weight=0`` 保留原 RRF 序。
    """
    combined = combined_scores(ce_scores, pool, weight)
    return sorted(range(len(pool)), key=lambda j: (-combined[j], j))


class Reranker:
    """交叉编码器重排：对召回池逐对打分，只影响排序、不改变召回集合。

    与 Embedder 同款惰性加载：未配置 ``rerank_model`` 时完全不 import
    onnxruntime 相关路径，开启重排的部署才付出模型常驻的内存代价。
    可注入假实现用于测试与评测对照。
    """

    def __init__(self, settings, engine=None):
        self.settings = settings
        self.model_name = settings.rerank_model
        self._engine = engine
        self._lock = threading.Lock()

    def load(self):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            raise RerankError("rerank_unavailable:uv_sync_extra_embed") from None
        cache = Path(self.settings.embedding_cache_dir or Path(self.settings.data_root) / "models")
        cache.mkdir(parents=True, exist_ok=True)
        try:
            return TextCrossEncoder(
                model_name=self.model_name,
                cache_dir=str(cache),
                threads=self.settings.embedding_threads or None,
            )
        except Exception as error:  # noqa: BLE001 - 下载失败、模型名写错、磁盘满都在这里
            raise RerankError("rerank_load_failed:" + type(error).__name__) from None

    def scores(self, query, documents):
        try:
            engine = self._engine
            if engine is None:
                with self._lock:
                    if self._engine is None:
                        self._engine = self.load()
                engine = self._engine
            return [float(s) for s in engine.rerank(query, list(documents))]
        except RerankError:
            raise
        except Exception as error:  # noqa: BLE001 - onnxruntime/tokenizers 异常类型不统一
            raise RerankError("rerank_failed:" + type(error).__name__) from None


class Embedder:
    """把文本变成归一化向量。

    三种来源：``local`` 在进程内跑 ONNX 小模型（无需外部服务），``ollama`` 与
    ``compatible`` 走 HTTP。三者共用同一个 ``key``，它是向量缓存的账本列——
    换模型或换地址都会让旧向量失效并触发重建，不会混用两套向量。
    """

    def __init__(self, settings):
        self.settings = settings
        self._engine = None
        self._lock = threading.Lock()
        self.key = json.dumps(
            ["local", settings.embedding_model]
            if settings.embedding_provider == "local"
            else [
                settings.embedding_provider,
                settings.embedding_base_url.rstrip("/"),
                settings.embedding_model,
            ]
        )

    def engine(self):
        """惰性加载：不开向量检索的部署不必多装 onnxruntime。"""
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = self.load()
        return self._engine

    def load(self):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise EmbeddingError("embedding_local_unavailable:uv_sync_extra_embed") from None
        settings = self.settings
        cache = Path(settings.embedding_cache_dir or Path(settings.data_root) / "models")
        cache.mkdir(parents=True, exist_ok=True)
        try:
            return TextEmbedding(
                model_name=settings.embedding_model,
                cache_dir=str(cache),
                threads=settings.embedding_threads or None,
            )
        except Exception as error:  # noqa: BLE001 - 首次下载失败、模型名写错、磁盘满都在这里
            raise EmbeddingError("embedding_local_load_failed:" + type(error).__name__) from None

    def local_embed(self, texts):
        engine = self.engine()
        try:
            with self._lock:
                # fastembed 给的是 numpy.float32，统一成 float 才能过后续的类型校验
                vectors = [[float(x) for x in vector] for vector in engine.embed(list(texts))]
        except Exception as error:  # noqa: BLE001 - onnxruntime/tokenizers 的异常类型不统一
            raise EmbeddingError("embedding_local_failed:" + type(error).__name__) from None
        return self.normalize(vectors, len(texts))

    def embed(self, texts):
        s = self.settings
        if s.embedding_provider == "disabled":
            raise EmbeddingError("embedding_disabled")
        if s.embedding_provider == "local":
            return self.local_embed(texts)
        path = "/api/embed" if s.embedding_provider == "ollama" else "/embeddings"
        payload = {"model": s.embedding_model, "input": texts}
        if s.embedding_provider == "ollama":
            payload["truncate"] = False
        headers = {}
        if s.embedding_api_key.get_secret_value():
            headers["Authorization"] = "Bearer " + s.embedding_api_key.get_secret_value()
        try:
            with httpx.Client(timeout=s.embedding_timeout_seconds) as client:
                response = client.post(
                    s.embedding_base_url.rstrip("/") + path, json=payload, headers=headers
                )
                response.raise_for_status()
                body = response.json()
            return self.decode(body, len(texts))
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            # Never return upstream bodies, request headers, keys or private text in errors.
            raise EmbeddingError("embedding_request_failed:" + type(error).__name__) from None

    def decode(self, body, count):
        if self.settings.embedding_provider == "ollama":
            vectors = body["embeddings"]
        else:
            rows = sorted(body["data"], key=lambda r: r["index"])
            if [row["index"] for row in rows] != list(range(count)):
                raise ValueError("invalid indices")
            vectors = [row["embedding"] for row in rows]
        return self.normalize(vectors, count)

    def normalize(self, vectors, count):
        if len(vectors) != count or not vectors:
            raise ValueError("invalid count")
        dimension = len(vectors[0])
        normalized = []
        for vector in vectors:
            if (
                not dimension
                or len(vector) != dimension
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector)
            ):
                raise ValueError("invalid vector")
            norm = math.sqrt(sum(x * x for x in vector))
            if not norm:
                raise ValueError("zero vector")
            normalized.append([x / norm for x in vector])
        return normalized

    async def embed_async(self, texts):
        s = self.settings
        if s.embedding_provider == "disabled":
            raise EmbeddingError("embedding_disabled")
        if s.embedding_provider == "local":
            try:
                return await asyncio.to_thread(self.local_embed, texts)
            except EmbeddingError as error:
                error.outcome = "failed"
                raise
        path = "/api/embed" if s.embedding_provider == "ollama" else "/embeddings"
        payload = {"model": s.embedding_model, "input": texts}
        if s.embedding_provider == "ollama":
            payload["truncate"] = False
        headers = {}
        if s.embedding_api_key.get_secret_value():
            headers["Authorization"] = "Bearer " + s.embedding_api_key.get_secret_value()
        try:
            async with httpx.AsyncClient(timeout=s.embedding_timeout_seconds) as client:
                response = await client.post(
                    s.embedding_base_url.rstrip("/") + path, json=payload, headers=headers
                )
                response.raise_for_status()
                return self.decode(response.json(), len(texts))
        except httpx.RequestError:
            error = EmbeddingError("embedding_request_failed:result_unknown")
            error.outcome = "unknown"
            raise error from None
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as cause:
            error = EmbeddingError("embedding_request_failed:" + type(cause).__name__)
            error.outcome = (
                "returned_invalid" if not isinstance(cause, httpx.HTTPError) else "failed"
            )
            raise error from None


class CourseIndex:
    def __init__(self, store):
        self.store = store

    def _build(self, material):
        space, mid = material["space_id"], material["id"]
        source = json.loads(material["source"])
        chunks = []
        provenance = {}
        if material["format"] in {"audio", "video"}:
            info = self.store.media(space, mid)
            provenance = {
                "transcript_source": info["transcript_source"],
                "transcript_state": info["transcript_state"],
            }

        def add(text, **position):
            if not text.strip():
                return
            if len(text) > 1200:
                for offset in range(0, len(text), 1000):
                    add(text[offset : offset + 1000], **position, excerpt_char_offset=offset)
                return
            ordinal = len(chunks)
            chunk = {
                "chunk_id": str(
                    uuid5(NAMESPACE_URL, f"{mid}:{material['revision']}:{INDEX_VERSION}:{ordinal}")
                ),
                "id": mid,
                "material_id": mid,
                "name": material["name"],
                "format": material["format"],
                "ordinal": ordinal,
                "excerpt": text,
                **provenance,
                **position,
            }
            if source:
                chunk["source"] = source
                if "start_seconds" in chunk:
                    offset = source.get("start_seconds", 0)
                    chunk["original_start_seconds"] = offset + chunk["start_seconds"]
                    chunk["original_end_seconds"] = offset + chunk["end_seconds"]
            if position.get("page"):
                chunk["page_url"] = (
                    f"/api/spaces/{space}/materials/{mid}/pages/{position['page']}/image"
                )
            if "start_seconds" in position:
                chunk["source_url"] = (
                    f"/#media={mid}&start={position['start_seconds']}&end={position['end_seconds']}"
                )
            chunks.append(chunk)

        if material["format"] in {"audio", "video"}:
            group = []
            for segment in self.store.segments(space, mid):
                if group and (
                    segment["end"] - group[0]["start"] > 45
                    or sum(len(s["text"]) for s in group) + len(segment["text"]) > 1000
                    or segment["start"] - group[-1]["end"] > 4
                ):
                    add(
                        "\n".join(s["text"] for s in group),
                        start_seconds=group[0]["start"],
                        end_seconds=max(s["end"] for s in group),
                    )
                    group = []
                # A single uploaded cue can exceed the regular chunk span. Keep its source interval honest.
                group.append(segment)
            if group:
                add(
                    "\n".join(s["text"] for s in group),
                    start_seconds=group[0]["start"],
                    end_seconds=max(s["end"] for s in group),
                )
        else:
            pages = (
                self.store.pages(space, mid) if material["format"] == "pdf" else [{"page": None}]
            )
            for page in pages:
                text = (
                    self.store.page(space, mid, page["page"])["text"]
                    if page["page"]
                    else material["text"]
                )
                lines = text.splitlines()
                start = 0
                while start < len(lines):
                    end, size = start, 0
                    while end < len(lines) and end - start < 18:
                        if end > start and (
                            size + len(lines[end]) > 1000 or lines[end].startswith("#")
                        ):
                            break
                        size += len(lines[end])
                        end += 1
                    add(
                        "\n".join(lines[start:end]),
                        page=page["page"],
                        start_line=start + 1,
                        end_line=end,
                    )
                    start = end
        return chunks

    def chunks(self, space_id, material_id=None):
        materials = self.store.materials(space_id)
        if material_id:
            self.store.material(space_id, material_id)
            materials = [m for m in materials if m["id"] == material_id]
        for item in materials:
            # Snapshot revision before reading mutable captions; validate again while publishing.
            for _ in range(2):
                material = self.store.material(space_id, item["id"])
                with self.store.connect() as db:
                    state = db.execute(
                        "SELECT revision,version FROM material_index WHERE material_id=?",
                        (item["id"],),
                    ).fetchone()
                if state and tuple(state) == (material["revision"], INDEX_VERSION):
                    break
                chunks = self._build(material)
                with self.store.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    revision = db.execute(
                        "SELECT revision FROM materials WHERE id=?", (item["id"],)
                    ).fetchone()[0]
                    if revision != material["revision"]:
                        continue
                    current = db.execute(
                        "SELECT revision,version FROM material_index WHERE material_id=?",
                        (item["id"],),
                    ).fetchone()
                    if current and tuple(current) == (revision, INDEX_VERSION):
                        break
                    db.execute(
                        "DELETE FROM chunk_vectors WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE material_id=?)",
                        (item["id"],),
                    )
                    db.execute("DELETE FROM chunks WHERE material_id=?", (item["id"],))
                    db.executemany(
                        "INSERT INTO chunks VALUES (?,?,?,?)",
                        [
                            (
                                c["chunk_id"],
                                item["id"],
                                c["ordinal"],
                                json.dumps(c, ensure_ascii=False),
                            )
                            for c in chunks
                        ],
                    )
                    db.execute(
                        "INSERT OR REPLACE INTO material_index VALUES (?,?,?)",
                        (item["id"], revision, INDEX_VERSION),
                    )
                break
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT c.payload FROM chunks c JOIN materials m ON m.id=c.material_id WHERE m.space_id=?"
                + (" AND m.id=?" if material_id else "")
                + " ORDER BY m.rowid,c.ordinal",
                (space_id, material_id) if material_id else (space_id,),
            )
            return [json.loads(row[0]) for row in rows]

    def context(self, space, chunk_id, before, after):
        chunks = self.chunks(space)
        anchor = next((c for c in chunks if c["chunk_id"] == chunk_id), None)
        if not anchor:
            raise LookupError("段落不存在或字幕已更新，请重新搜索")
        neighbors = [
            c
            for c in chunks
            if c["id"] == anchor["id"]
            and anchor["ordinal"] - before <= c["ordinal"] <= anchor["ordinal"] + after
        ]
        return {
            "anchor_chunk_id": chunk_id,
            "chunks": neighbors,
            "detail": "仅返回这些段落原文；PDF公式/图表需另行看图，视频需按返回时间抽样或回听。",
        }

    def outline(self, space, material_id, start, limit):
        material = self.store.material(space, material_id)
        chunks = self.chunks(space, material_id)
        entries = {}
        for c in chunks:
            key = c.get("page") or (
                int(c["start_seconds"] // 300) if "start_seconds" in c else c["ordinal"]
            )
            if key not in entries:
                entries[key] = {
                    k: v for k, v in c.items() if k not in {"excerpt", "source_url", "page_url"}
                }
                entries[key]["label"] = c["excerpt"].strip().splitlines()[0][:100]
            if "end_seconds" in c:
                entries[key]["end_seconds"] = c["end_seconds"]
        if material["format"] == "pdf":
            pages = {page["page"]: page for page in self.store.pages(space, material_id)}
            for number, page in pages.items():
                entry = entries.get(number)
                if entry is None:
                    entries[number] = {
                        "page": number,
                        "chunk_id": None,
                        "label": "无可提取文字，需查看页面图像"
                        if page["text_source"] == "none"
                        else "识别后仍无可读文字，需查看页面图像",
                    }
                    entry = entries[number]
                # The reader should see which pages are scans before spending a
                # tool call on them; recognition is per-page and only on demand.
                entry["text_source"] = page["text_source"]
            entries = dict(sorted(entries.items()))
        elif material["format"] in {"audio", "video"}:
            info = self.store.media(space, material_id)
            for key in range(math.ceil(info["duration"] / 300)):
                if key not in entries:
                    entries[key] = {
                        "start_seconds": key * 300,
                        "end_seconds": min((key + 1) * 300, info["duration"]),
                        "chunk_id": None,
                        "label": "此时间组无可检索字幕，可按需抽样查看画面",
                    }
            entries = dict(sorted(entries.items()))
        rows = list(entries.values())
        return {
            "material_id": material_id,
            "name": material["name"],
            "entries": rows[start : start + limit],
            "total": len(rows),
            "next_start": start + limit if start + limit < len(rows) else None,
            "navigation_kind": "page_first_line"
            if material["format"] == "pdf"
            else "five_minute_groups"
            if material["format"] in {"audio", "video"}
            else "text_sections",
            "detail": "目录标签来自原文首行；时间分组不是模型识别的语义章节，也不代表已读取对应完整内容。",
        }


class Retriever:
    def __init__(self, store, settings=None, embedder=None):
        self.store = store
        self.settings = settings or Settings(_env_file=None)
        self.index = CourseIndex(store)
        self.embedder = embedder or Embedder(self.settings)
        self.reranker = Reranker(self.settings) if self.settings.rerank_model else None
        self.lock = threading.Lock()
        self.lexical_lock = threading.Lock()
        self.lexical_index = {}
        self.jobs = {}
        self.queue = Jobs(store)

    def lexical_scores(self, chunks, query):
        """同一份语料只建一次 BM25 统计；块集合变了（上传、重新分块）才重建。"""
        signature = tuple(chunk["chunk_id"] for chunk in chunks)
        with self.lexical_lock:
            index = self.lexical_index.get(signature)
        if index is None:
            index = Bm25Index([chunk["excerpt"] for chunk in chunks])
            with self.lexical_lock:
                # 语料版本切换不会很频繁，保留最近几份足够；再多只是白占内存。
                while len(self.lexical_index) >= 4:
                    self.lexical_index.pop(next(iter(self.lexical_index)))
                self.lexical_index[signature] = index
        return index.scores(query)

    def vectors(self, chunks):
        ids = {c["chunk_id"] for c in chunks}
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT chunk_id,vector FROM chunk_vectors WHERE encoder=?", (self.embedder.key,)
            )
            return {r[0]: json.loads(r[1]) for r in rows if r[0] in ids}

    def status(self, space):
        chunks = self.index.chunks(space)
        count = len(self.vectors(chunks))
        with self.store.connect() as db:
            latest = db.execute(
                "SELECT state,error,config,input_version FROM jobs WHERE space_id=? AND kind='index' ORDER BY created DESC LIMIT 1",
                (space,),
            ).fetchone()
        state = self.jobs.get(space, {})
        if (
            latest
            and json.loads(latest["config"]) == configuration("index", self.settings)
            and latest["input_version"] == self.queue.version("index", space)
        ):
            state = dict(latest)
        return {
            "chunks": len(chunks),
            "embedded": count,
            "provider": self.settings.embedding_provider,
            "model": self.settings.embedding_model,
            "state": state.get("state")
            if state.get("state")
            in {"queued", "running", "paused", "failed", "cancelled", "disabled"}
            else "ready"
            if count == len(chunks) and chunks
            else "pending",
            "error": state.get("error"),
            "worker_online": self.queue.online(),
            "dense_ready": bool(chunks)
            and count == len(chunks)
            and self.settings.embedding_provider != "disabled",
        }

    def prepare(self, space, observer=None, raise_errors=False):
        # The worker serializes jobs; direct script callers retain vector-cache reuse.
        with self.lock:
            self.jobs[space] = {"state": "running"}
            try:
                chunks = self.index.chunks(space)
                vectors = self.vectors(chunks)
                pending = [c for c in chunks if c["chunk_id"] not in vectors]
                if self.settings.embedding_provider == "disabled":
                    self.jobs[space] = {"state": "disabled"}
                    return
                for start in range(0, len(pending), 16):
                    batch = pending[start : start + 16]
                    if observer:
                        observer("before", len(vectors) + start, len(chunks), batch)
                    embedded = self.embedder.embed([c["excerpt"] for c in batch])
                    with self.store.connect() as db:
                        db.execute("BEGIN IMMEDIATE")
                        for chunk, vector in zip(batch, embedded):
                            # Caption updates may have invalidated this batch during inference.
                            if db.execute(
                                "SELECT 1 FROM chunks WHERE chunk_id=?", (chunk["chunk_id"],)
                            ).fetchone():
                                db.execute(
                                    "INSERT OR REPLACE INTO chunk_vectors VALUES (?,?,?)",
                                    (chunk["chunk_id"], self.embedder.key, json.dumps(vector)),
                                )
                        if observer:
                            observer(
                                "saved",
                                len(vectors) + start + len(batch),
                                len(chunks),
                                [c["chunk_id"] for c in batch],
                                db,
                            )
                self.jobs[space] = {"state": "ready"}
            except EmbeddingError as error:
                self.jobs[space] = {"state": "failed", "error": str(error)}
                if raise_errors:
                    raise

    def apply_rerank(self, query, chunks, ranking):
        """对候选池前 N 名做交叉编码器重排；失败时如实降级回原排序。

        重排只决定"哪几个候选进前 5"，不扩大召回；返回 (新排序, 交叉编码器分数,
        状态)。``rerank_status``：``off`` 未配置、``ready`` 已重排、
        ``rerank_failed:*`` 降级原因——降级不能假装重排发生过。
        排序由 ``rerank_ce_weight`` 控制：默认 0.9 做 CE 概率与 RRF 的
        分数空间融合（见 :func:`fuse_ce_with_rrf` 的实验依据）；1.0 为纯 CE 替换。
        """
        if not self.reranker or self.settings.rerank_candidates <= 0:
            return ranking, {}, "off" if not self.settings.rerank_model else "disabled"
        pool = ranking[: self.settings.rerank_candidates]
        if not pool:
            return ranking, {}, "empty"
        try:
            scores = self.reranker.scores(query, [chunks[i]["excerpt"] for i in pool])
        except RerankError as error:
            return ranking, {}, str(error)
        if self.settings.rerank_ce_weight >= 1.0:
            # 纯 CE 替换：排序按原始分，展示值取 sigmoid（单调等价，值域 0-1）。
            order = sorted(range(len(pool)), key=lambda j: (-scores[j], pool[j]))
            values = {j: sigmoid(scores[j]) for j in range(len(pool))}
        else:
            values = combined_scores(scores, pool, self.settings.rerank_ce_weight)
            order = sorted(range(len(pool)), key=lambda j: (-values[j], pool[j]))
        return (
            [pool[j] for j in order] + list(ranking[self.settings.rerank_candidates :]),
            # 返回的 score 与排序键必须是同一个数：原始 CE logit 是负数（实测 -10 到 +7），
            # 而只返回 CE 概率时又会出现"分数大的排在后面"。融合分两者都避开，
            # 且没有校准，不能当拒答阈值用。
            {pool[j]: values[j] for j in range(len(pool))},
            "ready",
        )

    def search(self, space, query, material_id=None, depth=5):
        """``depth`` 是返回条数；评测做消融时才需要看第 6 名以后的候选。"""
        chunks = self.index.chunks(space, material_id)
        chunks, search_query, scoped_ids = course_scope(query, chunks)
        if not chunks:
            return {"hits": [], "retrieval": "bm25", "dense_status": "no_searchable_text"}
        scores = self.lexical_scores(chunks, search_query)
        lexical = sorted(
            (i for i, score in enumerate(scores) if score > 0), key=lambda i: (-scores[i], i)
        )
        ranking, fused = lexical, {}
        mode, dense_status = "bm25", "not_requested"
        dense_ranks = {}
        if self.settings.retrieval_mode == "hybrid":
            vectors = self.vectors(chunks)
            dense_status = (
                "disabled" if self.settings.embedding_provider == "disabled" else "index_incomplete"
            )
            if len(vectors) == len(chunks) and self.settings.embedding_provider != "disabled":
                try:
                    vector = getattr(self, "query_embed", self.embedder.embed)([search_query])[0]
                    if any(len(v) != len(vector) for v in vectors.values()):
                        raise EmbeddingError(
                            "embedding_dimension_changed:rebuild_with_new_model_id"
                        )
                    similarities = [
                        sum(a * b for a, b in zip(vector, vectors[c["chunk_id"]])) for c in chunks
                    ]
                    dense = sorted(range(len(chunks)), key=lambda i: (-similarities[i], i))[:30]
                    dense_ranks = {i: rank for rank, i in enumerate(dense, 1)}
                    ranking, fused = fuse(lexical[:30], dense)
                    mode, dense_status = "hybrid_rrf", "ready"
                except EmbeddingError as error:
                    dense_status = str(error)
        hits, per_page = [], Counter()
        ranking, ce_scores, rerank_status = self.apply_rerank(search_query, chunks, ranking)
        if ce_scores:
            mode = mode + "+rerank"
        lexical_ranks = {i: rank for rank, i in enumerate(lexical, 1)}
        for i in ranking:
            chunk = chunks[i]
            key = (chunk["id"], chunk.get("page") or chunk["ordinal"])
            if per_page[key] >= 2:
                continue
            per_page[key] += 1
            hits.append(
                {
                    **chunk,
                    "score": ce_scores.get(i, fused.get(i, scores[i])),
                    "bm25_rank": lexical_ranks.get(i),
                    "dense_rank": dense_ranks.get(i),
                }
            )
            if len(hits) == depth:
                break
        return {
            "hits": hits,
            "retrieval": mode,
            "dense_status": dense_status,
            "rerank_status": rerank_status,
            "scope": {
                "material_ids": scoped_ids,
                "query": search_query,
                "reason": "explicit_course_identifier" if scoped_ids else "caller_or_space",
            },
            "detail": "排名是候选线索，不是正确性置信度。可改写查询、限定资料、读取相邻段落或查看目录。",
        }
