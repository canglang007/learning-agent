"""Conversation-scoped learning observations, separate from claims of mastery."""

from typing import Literal

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


class Explanation(LearningNote):
    answer: str = Field(min_length=1, max_length=16000)


class FeedbackError(ValueError):
    pass


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
默认先给当前卡点的核心解释，接着给必要的一段推导或一个例子，避免拓展到无关章节。
一般首答尽量600中文字以内，局部追问尽量350字以内；完整推导可按需要展开，不截断公式、不牺牲正确性。用户指定详略时优先遵循。
讲解结束后不固定附加总结、练习题或重复回放清单；按当前需要提供来源与后续选择。"""
