from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TurnInput(Spec):
    question: str = Field(min_length=1, max_length=6000)
    focus: dict | None = None
    feedback: dict | None = None


class Target(Spec):
    asset: str
    pages: list[int] = []
    seconds: list[float] = []
    anchors: list[float] = []
    time_basis: Literal["clip", "original"] = "clip"
    modality: Literal["text", "image", "joint", "any"] = "any"

    @model_validator(mode="after")
    def bounds(self):
        if any(p < 1 for p in self.pages) or (
            self.seconds and (len(self.seconds) != 2 or not 0 <= self.seconds[0] < self.seconds[1])
        ):
            raise ValueError("参考位置需要物理页码或有效时间范围")
        return self


class Gold(Spec):
    required_concepts: list[str] = []
    forbidden_claims: list[str] = []
    evidence_targets: list[Target] = []
    review_required: bool = True
    numerical_checks: list[dict] = []
    state_equals: dict = {}
    memory_selected: bool | None = None


class Action(Spec):
    kind: Literal[
        "learning",
        "memory_create",
        "memory_attach",
        "memory_edit",
        "memory_forget",
        "memory_disable",
        "history",
    ]
    values: dict = {}


class Turn(Spec):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    input: TurnInput
    gold: Gold = Gold()
    conversation: str = "main"
    actions: list[Action] = []


class Case(Spec):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    family: str
    split: Literal["dev", "holdout"] = "dev"
    tags: list[str] = []
    turns: list[Turn] = Field(min_length=1)
    settings: dict = {}


class Suite(Spec):
    id: str
    version: str = "1"
    rubric_version: str = "1"
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def identities(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("重复case ID")
        families = {}
        for case in self.cases:
            if len({t.id for t in case.turns}) != len(case.turns):
                raise ValueError("重复turn ID")
            if case.family in families and families[case.family] != case.split:
                raise ValueError("同一概念族不能跨dev/holdout")
            families[case.family] = case.split
        return self
