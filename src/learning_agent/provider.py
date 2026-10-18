import httpx

from learning_agent.config import Settings


class ModelError(Exception):
    """Sanitized provider failure; do not expose upstream bodies or credentials."""


class OpenAIProvider:
    def __init__(self, settings: Settings, transport=None):
        self.settings, self.transport = settings, transport

    async def step(self, history, instructions, tools, allow_tools=True):
        settings = self.settings
        common = {"model": settings.model}
        if settings.protocol == "chat_completions":
            messages = [
                {k: v for k, v in item.items() if k != "provider_items"} for item in history
            ]
            payload = {
                **common,
                "messages": [{"role": "system", "content": instructions}, *messages],
                "tools": [{"type": "function", "function": tool} for tool in tools],
                "tool_choice": "auto" if allow_tools else "none",
            }
            endpoint = "/chat/completions"
        else:
            items = []
            for message in history:
                if message.get("provider_items") is not None:
                    items.extend(message["provider_items"])
                elif message["role"] == "tool":
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": message["tool_call_id"],
                            "output": message["content"],
                        }
                    )
                else:
                    items.append({"role": message["role"], "content": message["content"]})
            payload = {
                **common,
                "instructions": instructions,
                "input": items,
                "store": False,
                "include": ["reasoning.encrypted_content"],
                "tools": [{"type": "function", **tool} for tool in tools],
                "tool_choice": "auto" if allow_tools else "none",
            }
            endpoint = "/responses"
        try:
            async with httpx.AsyncClient(
                transport=self.transport, timeout=settings.timeout_seconds
            ) as client:
                response = await client.post(
                    settings.base_url.rstrip("/") + endpoint,
                    headers={"Authorization": "Bearer " + settings.api_key.get_secret_value()},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as error:
            raise ModelError(
                f"模型服务返回 HTTP {error.response.status_code}，请检查协议、型号或额度。"
            ) from None
        except (httpx.RequestError, ValueError):
            raise ModelError("模型服务连接失败、超时或返回了无效 JSON。") from None
        try:
            if settings.protocol == "chat_completions":
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ModelError("模型输出被截断，本轮未完成。")
                raw = choice["message"]
                message = {"role": "assistant", "content": raw.get("content") or ""}
                if raw.get("tool_calls"):
                    message["tool_calls"] = raw["tool_calls"]
            else:
                if data.get("status") != "completed":
                    raise ModelError("模型响应未完成，本轮未完成。")
                output = data["output"]
                text = "\n".join(
                    part.get("text", part.get("refusal", ""))
                    for item in output
                    if item["type"] == "message"
                    for part in item.get("content", [])
                    if part["type"] in {"output_text", "refusal"}
                )
                calls = [
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {"name": item["name"], "arguments": item["arguments"]},
                    }
                    for item in output
                    if item["type"] == "function_call"
                ]
                message = {"role": "assistant", "content": text, "provider_items": output}
                if calls:
                    message["tool_calls"] = calls
            for call in message.get("tool_calls", []):
                if (
                    not isinstance(call["id"], str)
                    or not isinstance(call["function"]["arguments"], str)
                    or not isinstance(call["function"]["name"], str)
                ):
                    raise TypeError()
            return message, data.get("usage", {})
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelError("模型响应不符合所选接口协议。") from None
