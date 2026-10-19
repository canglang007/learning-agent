"""Bounded provider inputs derived from, never substituted for, the original archive."""

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from urllib.parse import urlsplit

from learning_agent.provider import ModelError, OpenAIProvider, asset_info


class ContextError(ModelError):
    def __init__(self, code, detail, snapshot=None):
        super().__init__(detail)
        self.code = code
        self.snapshot = snapshot or {}


def profile_id(settings):
    # Namespacing only: never persist the URL's credentials/query or the key.
    service = hashlib.sha256(settings.base_url.encode()).hexdigest()[:16]
    return {
        "service": service,
        "host": urlsplit(settings.base_url).hostname,
        "model": settings.model,
        "protocol": settings.protocol,
    }


@dataclass
class TurnInput:
    question: str
    learning: dict = field(default_factory=dict)
    feedback: dict | None = None
    target: dict | None = None
    focus: dict | None = None

    def message(self, excerpt=False):
        content = self.question
        if (
            self.learning.get("working_note")
            or self.learning.get("feedback")
            or self.learning.get("state")
        ):
            content += (
                "\n[会话学习工作记录，仅供判断、可被用户修正："
                + json.dumps(self.learning, ensure_ascii=False)
                + "]"
            )
        if self.feedback:
            target = copy.deepcopy(self.target)
            if excerpt and target and self.feedback.get("focus_text"):
                text, focus = target["answer"], self.feedback["focus_text"]
                if text.count(focus) == 1:
                    start = text.index(focus)
                    left = text.rfind("\n\n", 0, start)
                    right = text.find("\n\n", start + len(focus))
                    target["answer"] = text[
                        left + 2 if left >= 0 else 0 : right if right >= 0 else len(text)
                    ]
                    target["truncated"] = target["answer"] != text
                    target["excerpt"] = True
            content += (
                "\n[本轮用户明确提交的学习反馈："
                + json.dumps({**self.feedback, "target": target}, ensure_ascii=False)
                + "]"
            )
        if self.focus:
            content += (
                "\n[当前查看位置，仅表示用户选中位置，不代表模型已查阅该位置："
                + json.dumps(self.focus, ensure_ascii=False)
                + "]"
            )
        return {"role": "user", "content": content}


@dataclass
class ProtocolGroup:
    id: str
    kind: str
    messages: list

    def dump(self):
        return asdict(self)


def validate_group(group):
    messages = group["messages"]
    if group["kind"] != "tools":
        if any(m.get("tool_calls") or m["role"] == "tool" for m in messages):
            raise ContextError("invalid_protocol", "历史协议分组不完整，本轮未发送模型请求。")
        return
    if not messages or messages[0]["role"] != "assistant":
        raise ContextError("invalid_protocol", "工具批次缺少助手调用，本轮未发送模型请求。")
    calls = messages[0].get("tool_calls", [])
    ids = [c["id"] for c in calls]
    results = [m.get("tool_call_id") for m in messages if m["role"] == "tool"]
    if not ids or len(set(ids)) != len(ids) or results != ids:
        raise ContextError("invalid_protocol", "工具调用与结果不配对，本轮未发送模型请求。")
    if any(m.get("tool_calls") for m in messages[1:]):
        raise ContextError("invalid_protocol", "工具批次边界不正确，本轮未发送模型请求。")
    native = messages[0].get("provider_items")
    if native is not None:
        native_ids = [p["call_id"] for p in native if p.get("type") == "function_call"]
        if native_ids != ids:
            raise ContextError(
                "invalid_protocol", "原生工具项目与调用ID不一致，本轮未发送模型请求。"
            )


def tool_index(group):
    """Deterministic observation locations, not an interpretation of source content."""
    entries = []
    calls = group["messages"][0].get("tool_calls", [])
    for call, result in zip(calls, [m for m in group["messages"] if m["role"] == "tool"]):
        try:
            observation = json.loads(result["content"])
        except (ValueError, TypeError):
            observation = {}
        entries.append(
            {
                "tool": call["function"]["name"],
                **{
                    k: observation[k]
                    for k in (
                        "material_id",
                        "name",
                        "page",
                        "region",
                        "start_line",
                        "end_line",
                        "start_seconds",
                        "end_seconds",
                        "error",
                    )
                    if k in observation
                },
            }
        )
    return {
        "role": "user",
        "content": "[历史工具索引，正文与图像已移出本次输入，"
        "不是当前查阅或来源结论；需要时重新读取："
        + json.dumps({"group_id": group["id"], "observations": entries}, ensure_ascii=False)
        + "]",
    }


@dataclass
class PreparedContext:
    messages: list
    snapshot: dict
    asset_ids: set


class ContextManager:
    def __init__(self, settings, builder=None):
        self.settings = settings
        self.builder = builder or OpenAIProvider(settings)
        self.profile = profile_id(settings)

    def prepare(
        self, prior_turns, current_groups, turn_input, instructions, tools, allow, summary=None
    ):
        blocks, omissions = [], []
        if summary:
            from learning_agent.summaries import summary_message

            blocks.append(
                {
                    "id": summary["id"],
                    "kind": "summary",
                    "messages": [summary_message(summary)],
                    "turn": "summary",
                    "pinned": True,
                    "mode": "summary",
                }
            )
        for turn in prior_turns:
            if summary and 0 < turn["turn_seq"] <= summary["covered_end"]:
                omissions.append(
                    {"group_id": f"turn:{turn['turn_seq']}", "reason": "covered_by_summary"}
                )
                continue
            compatible = turn["profile"] == self.profile and turn["replay_status"] == "ready"
            if not compatible:
                blocks.append(
                    {
                        "id": f"visible:{turn['turn_seq']}",
                        "kind": "visible",
                        "messages": turn["visible"],
                        "turn": turn["turn_seq"],
                        "pinned": False,
                        "mode": "visible",
                    }
                )
                if turn["groups"]:
                    omissions.append(
                        {
                            "group_id": f"turn:{turn['turn_seq']}",
                            "reason": "legacy_or_profile_change_or_failed",
                        }
                    )
                continue
            for group in copy.deepcopy(turn["groups"]):
                validate_group(group)
                group.update(turn=turn["turn_seq"], pinned=False, mode="full")
                blocks.append(group)
            if turn.get("delivered_answer"):
                blocks.append(
                    {
                        "id": f"delivery:{turn['turn_seq']}",
                        "kind": "visible",
                        "messages": [
                            {
                                "role": "user",
                                "content": "[上一轮实际交付的正文，来自已保存答案："
                                + turn["delivered_answer"]
                                + "]",
                            }
                        ],
                        "turn": turn["turn_seq"],
                        "pinned": False,
                        "mode": "visible",
                    }
                )
        for i, source in enumerate(current_groups):
            group = copy.deepcopy(source.dump())
            validate_group(group)
            group.update(
                turn="current", pinned=(i == 0 or i == len(current_groups) - 1), mode="full"
            )
            blocks.append(group)

        # Current observations take precedence; then focus/feedback and recency.
        images = []
        for b in blocks:
            for m in b["messages"]:
                for a in m.get("attachments", []):
                    images.append((b, m, a))
        focus_material = (turn_input.focus or {}).get("material_id")
        feedback_run = (turn_input.feedback or {}).get("target_run_id")
        images = [
            item
            for _, item in sorted(
                enumerate(images),
                key=lambda pair: (
                    pair[1][0]["pinned"],
                    focus_material is not None
                    and focus_material in pair[1][2].get("asset_path", ""),
                    feedback_run is not None and feedback_run in pair[1][0]["id"],
                    pair[0],
                ),
                reverse=True,
            )
        ]
        selected = set()
        for b, m, a in images:
            reason = None
            try:
                asset_info(self.settings.data_root, a)
            except ModelError:
                reason = "image_unavailable"
            if not self.settings.enable_vision:
                reason = "vision_disabled"
            if a["asset_path"] in selected:
                reason = "image_already_attached"
            if reason is None and len(selected) >= self.settings.max_context_images:
                reason = "image_limit"
            if reason:
                m["attachments"] = [x for x in m.get("attachments", []) if x is not a]
                m["content"] += (
                    "\n[该图像未附在本次输入：" + reason + "；不能声称当前看见图中细节。]"
                )
                omissions.append({"group_id": b["id"], "reason": reason})
                b["mode"] = "without_old_images"
            else:
                selected.add(a["asset_path"])

        # Some compatible gateways reuse call IDs between turns. Avoid ambiguous replay.
        used = set()
        for b in reversed(blocks):
            ids = {c["id"] for m in b["messages"] for c in m.get("tool_calls", [])}
            if ids & used:
                if b["pinned"]:
                    raise ContextError(
                        "invalid_protocol", "当前工具调用ID重复，本轮未发送模型请求。"
                    )
                carriers = [m for m in b["messages"] if m.get("attachments")]
                b["messages"], b["mode"] = [tool_index(b), *carriers], "index"
                omissions.append({"group_id": b["id"], "reason": "call_id_reused"})
            else:
                used.update(ids)

        def materialize():
            messages = [m for b in blocks for m in b["messages"]]
            if omissions:
                messages = [
                    {
                        "role": "user",
                        "content": "[系统上下文状态：本次仅使用部分历史；旧索引和已移出图像不代表当前已读取。]",
                    },
                    *messages,
                ]
            return messages

        def measure():
            messages = materialize()
            try:
                request = self.builder.build_request(
                    messages, instructions, tools, allow, encode_images=False
                )
                return messages, self.builder.estimate(request)
            except (ValueError, TypeError):
                raise ContextError(
                    "invalid_protocol", "模型请求包含无效协议数据，未发送。"
                ) from None

        def fits(estimate):
            s = self.settings
            return (
                estimate["input_total"]
                <= s.context_budget_tokens
                - s.context_output_reserve_tokens
                - s.context_headroom_tokens
                and estimate["request_bytes"] <= s.context_max_request_bytes
            )

        def snapshot(estimate):
            return {
                "schema_version": 1,
                "summary_id": summary["id"] if summary else None,
                "profile": self.profile,
                "estimate": estimate,
                "input_budget": self.settings.context_budget_tokens
                - self.settings.context_output_reserve_tokens
                - self.settings.context_headroom_tokens,
                "selection": [{"group_id": b["id"], "mode": b["mode"]} for b in blocks],
                "omissions": omissions,
            }

        messages, estimate = measure()
        if not fits(estimate):
            for b in blocks:
                if b["pinned"]:
                    continue
                changed = False
                for m in b["messages"]:
                    if m.get("attachments"):
                        m.pop("attachments")
                        m["content"] += "\n[历史图像已移出本次输入，需要细节请重新查看。]"
                        changed = True
                if changed:
                    b["mode"] = (
                        "index_without_images" if b["mode"] == "index" else "without_old_images"
                    )
                    omissions.append({"group_id": b["id"], "reason": "image_budget_pressure"})
                    messages, estimate = measure()
                    if fits(estimate):
                        break
        if not fits(estimate):
            for b in blocks:
                if b["kind"] == "tools" and not b["pinned"] and not b["mode"].startswith("index"):
                    b["messages"], b["mode"] = [tool_index(b)], "index"
                    omissions.append({"group_id": b["id"], "reason": "tool_budget_pressure"})
                    messages, estimate = measure()
                    if fits(estimate):
                        break
        if not fits(estimate):
            for turn in dict.fromkeys(b["turn"] for b in blocks):
                optional = [b for b in blocks if b["turn"] == turn and not b["pinned"]]
                for b in optional:
                    blocks.remove(b)
                    omissions.append({"group_id": b["id"], "reason": "history_budget_pressure"})
                messages, estimate = measure()
                if fits(estimate):
                    break
        if not fits(estimate) and turn_input.feedback:
            for b in blocks:
                if b["kind"] == "input" and b["turn"] == "current":
                    b["messages"] = [turn_input.message(excerpt=True)]
                    omissions.append({"group_id": b["id"], "reason": "feedback_excerpt"})
            messages, estimate = measure()
        if not fits(estimate) and summary:
            blocks = [b for b in blocks if b["kind"] != "summary"]
            omissions.append({"group_id": summary["id"], "reason": "summary_budget_pressure"})
            messages, estimate = measure()
        if not fits(estimate):
            raise ContextError(
                "context_required_too_large",
                "当前问题、反馈目标或最新工具观察超过上下文预算。请选中具体步骤、缩短资料片段，"
                "或调整本机上下文配置；原记录仍保留。",
                snapshot(estimate),
            )
        assets = {a["asset_path"] for m in messages for a in m.get("attachments", [])}
        return PreparedContext(messages, snapshot(estimate), assets)
