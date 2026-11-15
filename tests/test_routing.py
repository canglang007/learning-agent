from client_helpers import TestClient, close_for

from learning_agent.app import create_app
from learning_agent.config import Settings
from learning_agent.routing import choose_route


class Plain:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def step(self, history, instructions, tools, allow_tools):
        self.calls.append(
            {
                "history": [dict(message) for message in history],
                "instructions": instructions,
                "tools": tools,
                "allow_tools": allow_tools,
            }
        )
        return {"role": "assistant", "content": self.answer}, {"total_tokens": 5}


def route_events(client, run_id):
    return client.get(f"/api/runs/{run_id}/events?after=0").json()


def ask(client, conversation_id, content):
    return client.post(
        f"/api/conversations/{conversation_id}/messages", json={"content": content}
    ).json()


def test_router_rules():
    assert choose_route("用简短例子讲清近似和严格相等的区别。") == (
        "direct",
        "general_question",
    )
    assert choose_route("CS336为什么归一化运算少但耗时多？请自主找课程材料。") == (
        "fixed_rag",
        "material_reference",
    )
    assert choose_route("请实际看cs336-architecture.pdf物理第10页，解释残差通路。") == (
        "agent",
        "explicit_visual_or_time",
    )
    assert choose_route("解释这个概念", focus={"page": 10}) == ("agent", "focus_present")
    assert choose_route("换个例子", feedback={"kind": "example"}) == (
        "agent",
        "feedback_present",
    )
    assert choose_route("0.17%与25.5%能相除吗？", followup=True) == (
        "agent",
        "followup_present",
    )
    assert choose_route("解释这个概念", policy="agent") == ("agent", "policy_agent")


def test_course_codes_come_from_material_names():
    # 课程标识只能来自工作区资料名，代码里不保留任何具体课程。
    names = ["cs336-architecture.pdf", "study-notes.md", "worksheet.pdf"]
    assert choose_route("CS336 为什么归一化运算少但耗时多？", material_names=names) == (
        "fixed_rag",
        "course_code_reference",
    )
    assert choose_route("cs336里归一化放在哪一步", material_names=names) == (
        "fixed_rag",
        "course_code_reference",
    )
    # 资料里没有该标识时不能凭空命中。
    assert choose_route("cs336里归一化放在哪一步") == ("direct", "general_question")
    assert choose_route("stat20里怎么讲残差", material_names=names) == (
        "direct",
        "general_question",
    )
    # 换一门课只需换资料名。
    assert choose_route(
        "stat20里怎么讲残差", material_names=["stat20-lecture3.pdf"]
    ) == ("fixed_rag", "course_code_reference")
    # 标识必须作为独立词出现，不能命中更长串的一部分。
    assert choose_route(
        "我的账号是xcs3367，这个怎么算", material_names=names
    ) == ("direct", "general_question")
    # 文件格式词不算课程标识。
    from learning_agent.routing import course_codes

    assert course_codes(["cs336-prenorm.mp4", "lecture-1080p.mov"]) == {"cs336"}
    # 通用资料词仍然生效，与课程标识无关。
    assert choose_route("这课件在讲什么", material_names=["worksheet.pdf"]) == (
        "fixed_rag",
        "material_reference",
    )


def test_instructions_require_material_id_lookup_before_reading():
    from learning_agent.agent import INSTRUCTIONS

    assert "list_materials" in INSTRUCTIONS
    assert "别拿文件名或猜的ID去试" in INSTRUCTIONS


def test_auto_route_direct_uses_no_tools_and_records_route(tmp_path):
    close_for(tmp_path)
    model = Plain("近似需要说明误差条件。")
    client = TestClient(
        create_app(
            Settings(_env_file=None, data_root=tmp_path, route_policy="auto"), model
        )
    )
    space = client.post("/api/spaces", json={"content": "学习"}).json()
    conversation = client.post(f"/api/spaces/{space['id']}/conversations").json()
    result = ask(client, conversation["id"], "用简短例子讲清近似和严格相等的区别。")

    assert result["state"] == "completed"
    assert model.calls[0]["tools"] == []
    events = route_events(client, result["run_id"])
    assert any(event["type"] == "route.selected" and event["payload"]["route"] == "direct" for event in events)


def test_auto_route_fixed_rag_injects_retrieval_and_records_evidence(tmp_path):
    close_for(tmp_path)
    model = Plain("归一化是在分支内做的。")
    client = TestClient(
        create_app(
            Settings(_env_file=None, data_root=tmp_path, route_policy="auto"), model
        )
    )
    space = client.post("/api/spaces", json={"content": "学习"}).json()
    client.post(
        f"/api/spaces/{space['id']}/materials",
        files={
            "file": (
                "notes.md",
                "CS336 的归一化在分支内完成，跳连不经过归一化。".encode(),
                "text/markdown",
            )
        },
    )
    conversation = client.post(f"/api/spaces/{space['id']}/conversations").json()
    result = ask(client, conversation["id"], "CS336课程材料里归一化怎么讲？")

    assert result["state"] == "completed"
    assert model.calls[0]["tools"] == []
    assert "资料片段" in model.calls[0]["instructions"]
    events = route_events(client, result["run_id"])
    assert any(
        event["type"] == "route.selected" and event["payload"]["route"] == "fixed_rag"
        for event in events
    )
    context = client.get(f"/api/runs/{result['run_id']}/context").json()
    assert any(snapshot.get("retrieval_evidence") for snapshot in context)


def test_auto_route_reads_course_code_from_workspace_materials(tmp_path):
    close_for(tmp_path)
    model = Plain("归一化在子层内部完成。")
    client = TestClient(
        create_app(
            Settings(_env_file=None, data_root=tmp_path, route_policy="auto"), model
        )
    )
    space = client.post("/api/spaces", json={"content": "学习"}).json()
    client.post(
        f"/api/spaces/{space['id']}/materials",
        files={
            "file": (
                "cs336-notes.md",
                "归一化在子层内部完成，跳连不经过归一化。".encode(),
                "text/markdown",
            )
        },
    )
    conversation = client.post(f"/api/spaces/{space['id']}/conversations").json()
    # 问题里没有任何通用“资料/课程/材料”词，只有资料名里的课程标识。
    result = ask(client, conversation["id"], "cs336里归一化在哪一步？")

    assert result["state"] == "completed"
    events = route_events(client, result["run_id"])
    assert any(
        event["type"] == "route.selected"
        and event["payload"]["route"] == "fixed_rag"
        and event["payload"]["reason"] == "course_code_reference"
        for event in events
    )


def test_auto_route_keeps_agent_for_visual_reference(tmp_path):
    close_for(tmp_path)
    model = Plain("需要实际查看页面。")
    client = TestClient(
        create_app(
            Settings(_env_file=None, data_root=tmp_path, route_policy="auto"), model
        )
    )
    space = client.post("/api/spaces", json={"content": "学习"}).json()
    conversation = client.post(f"/api/spaces/{space['id']}/conversations").json()
    result = ask(client, conversation["id"], "请实际看cs336-architecture.pdf第10页，解释残差通路。")

    assert result["state"] == "completed"
    assert model.calls[0]["tools"]
    events = route_events(client, result["run_id"])
    assert any(event["type"] == "route.selected" and event["payload"]["route"] == "agent" for event in events)
