import re


def check(name, status, reason, refs=()):
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "evidence_refs": list(refs),
        "grader": "rules-v1",
    }


def matches(entry, target, assets, image=False):
    asset = assets.get(target["asset"], {})
    mid = entry.get("material_id", entry.get("id"))
    path = entry.get("asset_path", "")
    if path:
        parts = path.split("/")
        mid = parts[1] if len(parts) > 2 and parts[0] == "materials" else None
    if mid != asset.get("id"):
        return False
    if target.get("pages"):
        page = entry.get("page")
        if path:
            found = re.search(r"/page-(\d+)-", path)
            page = int(found[1]) if found else None
        return page in target["pages"]
    offset = (
        asset.get("source_start_seconds", 0)
        if target.get("time_basis", "clip") == "original"
        else 0
    )
    if target.get("time_basis") == "original" and "source_start_seconds" not in asset:
        return None
    anchors = target.get("anchors", [])
    interval = target.get("seconds", [])
    if anchors or interval:
        if image:
            return "seconds" in entry and (
                not interval or interval[0] <= entry["seconds"] + offset <= interval[1]
            )
        if "start_seconds" not in entry or "end_seconds" not in entry:
            return False
        a, b = entry["start_seconds"] + offset, entry["end_seconds"] + offset
        return all(a <= point <= b for point in anchors) and (
            not interval or a <= interval[0] < interval[1] <= b
        )
    return True


def grade(trace, gold, assets):
    checks, findings = [], []
    assistant = [m for m in trace.get("messages", []) if m["role"] == "assistant"]
    completed = trace.get("state") in {"completed", "budget_exhausted"}
    checks.append(
        check(
            "runtime_delivery",
            "pass" if completed and len(assistant) == 1 and trace.get("answer") else "fail",
            "仅统计canonical答案，草稿不参与",
            ["messages"],
        )
    )
    if not completed:
        findings.append(
            {
                "category": "provider_runtime",
                "status": "observed",
                "reason": trace.get("error") or trace.get("recovery_reason") or trace.get("state"),
                "evidence_refs": ["state", "operations"],
            }
        )
    manifests = [
        (op["id"], op["input_ref"])
        for op in trace.get("operations", [])
        if op["purpose"] == "main_model" and op.get("input_ref", {}).get("schema_version") == 1
    ]
    evidence = [(oid, e) for oid, m in manifests for e in m.get("evidence", [])]
    images = [(oid, e) for oid, m in manifests for e in m.get("images", [])]
    for i, target in enumerate(gold.get("evidence_targets", [])):
        modality = target.get("modality", "any")
        text_hits = [
            oid
            for oid, e in evidence
            if matches(e, target, assets)
            or any(matches(h, target, assets) for h in e.get("hits", []))
        ]
        image_hits = [oid for oid, e in images if matches(e, target, assets, True)]
        passed = (
            bool(image_hits)
            if modality == "image"
            else bool(image_hits and text_hits)
            if modality == "joint"
            else bool(text_hits or image_hits)
        )
        strict = modality in {"image", "joint"}
        checks.append(
            check(
                f"evidence_{i}",
                "pass" if passed else "fail" if manifests and strict else "needs_review",
                "模型输入中的证据位置匹配；内容支持关系仍需复核",
                text_hits + image_hits,
            )
        )
        if not passed:
            findings.append(
                {
                    "category": "context_selection"
                    if any(
                        matches(t.get("observation", {}), target, assets)
                        for t in trace.get("tools", [])
                    )
                    else "tool_selection",
                    "status": "suspected",
                    "reason": "所需证据未在尝试输入清单匹配；需检查替代来源、工具观察和上下文省略",
                    "evidence_refs": ["operations.input_ref", "tools"],
                }
            )
    for key, expected in gold.get("state_equals", {}).items():
        actual = (trace.get("learning_state") or {}).get(key)
        if isinstance(actual, dict) and isinstance(expected, str):
            actual = actual.get("text")
        checks.append(
            check(
                "state:" + key,
                "pass" if actual == expected else "fail",
                f"学习状态字段应为{expected!r}，实际{actual!r}",
                ["learning_state." + key],
            )
        )
    if gold.get("memory_selected") is not None:
        actual = any(m.get("memory", {}).get("selected") for _oid, m in manifests)
        checks.append(
            check(
                "memory_selected",
                "pass"
                if manifests and actual == gold["memory_selected"]
                else "fail"
                if manifests
                else "needs_review",
                "根据实际尝试注入记忆清单判断",
                [oid for oid, _m in manifests],
            )
        )
    for i, spec in enumerate(gold.get("numerical_checks", [])):
        try:
            found = re.search(spec["pattern"], trace.get("answer") or "")
            good = found and abs(float(found.group("value")) - spec["value"]) <= spec.get(
                "tolerance", 0
            )
            checks.append(
                check(
                    f"numeric_{i}",
                    "pass" if good else "needs_review",
                    "显式数值格式/容差检查；未匹配可能是等效表达，需复核",
                    ["answer"],
                )
            )
        except (KeyError, ValueError, TypeError, re.error):
            checks.append(
                check(
                    f"numeric_{i}",
                    "needs_review",
                    "grader_error：数值检查定义无效，不当成答案错误",
                    ["answer"],
                )
            )
    if gold.get("review_required", True):
        checks.append(
            check(
                "explanation_quality",
                "needs_review",
                "准确性、卡点、清晰度、反馈适应和不支持的断言需要有理由复核",
            )
        )
    for tool in trace.get("tools", []):
        observation = tool.get("observation", {})
        if observation.get("error"):
            findings.append(
                {
                    "category": "provider_runtime",
                    "status": "observed",
                    "reason": observation["error"],
                    "evidence_refs": ["tools"],
                }
            )
    return {"schema_version": 1, "checks": checks, "findings": findings}
