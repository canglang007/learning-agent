"""Conversation-scoped learning observations, separate from claims of mastery."""

import copy
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["stuck", "example", "restate", "understood"]
    target_run_id: str = Field(min_length=1, max_length=80)
    focus_text: str = Field(default="", max_length=1000)


class LearningNote(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    topic: str = Field(min_length=1, max_length=120)
    confusion: str = Field(max_length=400)
    approach: Literal[
        "clarify",
        "prerequisite",
        "step_by_step",
        "example",
        "analogy",
        "source_check",
        "check_restatement",
        "explain",
        "acknowledge",
    ]
    reason: str = Field(min_length=1, max_length=300)


class StateUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    goal: str | None = Field(default=None, max_length=240)
    active_topic: str | None = Field(default=None, max_length=120)
    confusion: str | None = Field(default=None, max_length=400)
    user_quote: str | None = Field(default=None, max_length=300)
    add_questions: list[str] = Field(default_factory=list, max_length=4)
    resolve_question_ids: list[str] = Field(default_factory=list, max_length=4)


class UserStateEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: int = Field(ge=0)
    goal: str | None = Field(default=None, max_length=240)
    goal_pinned: bool | None = None
    active_topic: str | None = Field(default=None, max_length=120)
    confusion: str | None = Field(default=None, max_length=400)
    add_questions: list[str] = Field(default_factory=list, max_length=4)
    close_question_ids: list[str] = Field(default_factory=list, max_length=12)
    reset: bool = False


class Explanation(LearningNote):
    answer: str = Field(min_length=1, max_length=16000)
    state_update: StateUpdate | None = None


class FeedbackError(ValueError):
    pass


class LearningConflict(ValueError):
    pass


def empty_state():
    return {
        "revision": 0,
        "context_epoch": 0,
        "goal": None,
        "goal_pinned": False,
        "active_topic": None,
        "confusion": None,
        "open_questions": [],
        "attempts": [],
        "evidence_refs": [],
        "reset": False,
    }


def state_for_context(state):
    """The UI keeps all records; the model gets current work plus bounded pointers."""
    return {
        **{
            k: state[k]
            for k in (
                "revision",
                "context_epoch",
                "goal",
                "goal_pinned",
                "active_topic",
                "confusion",
                "reset",
            )
        },
        "open_questions": [
            {k: q[k] for k in ("id", "text", "source_run_id", "topic_id")}
            for q in state["open_questions"]
            if q["status"] == "open"
        ],
        "evidence_refs": state["evidence_refs"][-3:],
    }


def apply_learning_update(
    base, patch, *, run_id=None, note=None, question="", feedback=None, user=False, evidence_refs=()
):
    """Validate evidence and scope before any persisted answer/state transaction."""
    state = copy.deepcopy(base)
    state.setdefault(
        "goal_pinned", bool(state["goal"] and state["goal"]["origin"] == "user_explicit")
    )
    origin = "user_explicit" if user else "agent_hypothesis"
    quote = patch.get("user_quote")
    if quote and quote not in question:
        raise ValueError("状态更新的user_quote必须是本轮用户原话")
    topic = state["active_topic"]
    old_feedback = (
        not user
        and feedback
        and topic
        and feedback["target_run_id"] != topic["source_run_id"]
        and note
        and note["approach"] == "acknowledge"
    )
    for key in ("goal", "active_topic", "confusion"):
        if key not in patch or patch[key] is None and not user or old_feedback:
            continue
        value = patch[key]
        if key == "goal" and not user and state["goal_pinned"]:
            continue
        old = state[key]
        # A previous direct correction cannot be overwritten by unquoted guesses.
        if (
            not user
            and old
            and old["origin"] == "user_explicit"
            and (not quote or value is None or value not in quote)
        ):
            continue
        if value is None or not value.strip():
            state[key] = None
            continue
        value = value.strip()
        if key == "active_topic":
            state[key] = {
                "id": old["id"] if old and old["text"] == value else str(uuid4()),
                "text": value,
                "origin": origin,
                "source_run_id": run_id,
                "user_quote": quote,
            }
        else:
            state[key] = {
                "text": value,
                "origin": origin,
                "source_run_id": run_id,
                "user_quote": quote,
            }
    if user and "goal" in patch:
        state["goal_pinned"] = bool(state["goal"])
    if user and patch.get("goal_pinned") is not None:
        state["goal_pinned"] = bool(state["goal"]) and patch["goal_pinned"]
    for text in patch.get("add_questions", []):
        if not text.strip() or len(text) > 240:
            raise ValueError("待解释问题需为1至240字符")
        if old_feedback or any(
            q["text"] == text.strip() and q["status"] == "open" for q in state["open_questions"]
        ):
            continue
        if sum(q["status"] == "open" for q in state["open_questions"]) >= 8:
            raise ValueError("未解决问题已达8项，请先合并或关闭已有问题")
        state["open_questions"].append(
            {
                "id": str(uuid4()),
                "text": text.strip(),
                "topic_id": state["active_topic"]["id"] if state["active_topic"] else None,
                "source_run_id": run_id,
                "origin": origin,
                "status": "open",
                "feedback_run_id": None,
            }
        )
    for qid in patch.get("close_question_ids" if user else "resolve_question_ids", []):
        item = next((q for q in state["open_questions"] if q["id"] == qid), None)
        if item is None:
            raise ValueError("待解释问题ID不属于当前学习状态")
        if not user and (
            not feedback
            or feedback["kind"] != "understood"
            or item["source_run_id"] != feedback["target_run_id"]
        ):
            raise ValueError("标记理解须有指向该问题来源回答的明确理解反馈；不能推断掌握")
        item["status"] = "closed_by_user" if user else "user_understood"
        item["feedback_run_id"] = run_id
    state["open_questions"] = [q for q in state["open_questions"] if q["status"] == "open"] + [
        q for q in state["open_questions"] if q["status"] != "open"
    ][-8:]
    if note:
        state["attempts"] = [
            *state["attempts"],
            {"run_id": run_id, "approach": note["approach"], "topic": note["topic"]},
        ][-8:]
    state["evidence_refs"] = [*state["evidence_refs"], *evidence_refs][-12:]
    state["revision"] += 1
    if user:
        state["context_epoch"] += 1
    return state


LEARNING_GUIDANCE = """用户反馈属于观察，不等于测验通过。工作笔记是你的可修正判断，不是用户掌握情况。
最终使用deliver_explanation一次性交付answer和本轮简短工作记录，不再另写重复的正文消息。
记录topic、当前confusion、实际采用的approach及简短reason；这是策略摘要，不写内部思维链，不给掌握分数。
topic写正在学习的知识概念，不用“确认反馈”等动作作为主题；确认理解时保留此前知识主题。
记录主题切换和卡点变化；用户未明确卡点时写尚待澄清。用户表示理解时用acknowledge，只确认反馈范围。
交付工具应独立调用，先取得其他工具观察再交付；工具预算用完无法调用时可直接给正文，并说明未完成部分。
用户说不懂时，先针对指出的步骤诊断；不确定卡点就简短澄清。若同一讲法没有帮助，改变表示方式或补前提，避免把原答案加长重述。
用户请求例子时围绕同一卡点构造可算的小例子；需要依赖资料的结论仍要核对来源。
用户复述时，逐点检查推理，指出具体正确处、错误处与缺失条件；即使选择了“我来复述”也不能预先判为正确。
用户明确说某一点明白了，可记为该回答的自述理解反馈；不推广成整个概念已掌握，也不强制出题或继续长篇讲课。
反馈可能指向此前某一回答，要看target_run_id及对应内容；旧回答反馈不能覆盖新主题。
可用state_update更新学习目标、当前主题、卡点或添加待解释问题；不变字段填null，列表不变填[]。
修改用户在面板明确填写的字段时，新值需逐字来自本轮user_quote；普通猜测不能覆盖明确更正。
goal_pinned=true时goal由用户固定，不得改写；临时换题仍按当前问题讲解，只改变active_topic。
这些是工作假设，可被用户修改。user_quote只能引用本轮用户确实说过的话；不捏造用户目标。
resolve_question_ids只用于有明确understood反馈且目标回答与该问题source_run_id相同的条目。
复述核对只记录实际检查与讲法，不自动关闭问题或推断掌握。确认旧回答时保持新主题和卡点。
当前学习状态优先于旧摘要；用户本轮要求和明确纠正优先。状态被用户清空后不要从旧记录恢复旧卡点。
历史摘要是二手记录，不是资料查阅；精确用户原话可recall_learning_context，资料归因仍需资料工具核对。
默认先给当前卡点的核心解释，接着给必要的一段推导或一个例子，避免拓展到无关章节。
一般首答尽量600中文字以内，局部追问尽量350字以内；完整推导可按需要展开，不截断公式、不牺牲正确性。用户指定详略时优先遵循。
讲解结束后不固定附加总结、练习题或重复回放清单；按当前需要提供来源与后续选择。"""
