"""Provider capability registry used before a model request is sent."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelCapabilities:
    text: bool = True
    vision: bool = False
    tool_calling: bool = False
    structured_output: bool = False
    streaming: bool = False


def capabilities(settings):
    """Return conservative capabilities for the configured provider.

    OpenAI-compatible gateways are deliberately marked by configuration and
    model name; an unknown model never receives an optimistic vision claim.
    """
    provider = settings.base_url.lower()
    model = settings.model.lower()
    vision = settings.enable_vision and any(
        token in model for token in ("vision", "omni", "gpt-4", "gpt-5", "gpt-6", "claude")
    )
    if "ollama" in provider or "11434" in provider:
        vision = settings.enable_vision and any(token in model for token in ("llava", "qwen2.5vl", "gemma3"))
    return ModelCapabilities(
        vision=vision,
        tool_calling=settings.protocol in {"chat_completions", "responses"},
        structured_output=settings.protocol in {"chat_completions", "responses"},
        streaming=True,
    )


def status(settings):
    value = asdict(capabilities(settings))
    value.update({"provider": settings.base_url, "model": settings.model})
    return value


def require(settings, *, vision=False, tools=False):
    value = capabilities(settings)
    if vision and not value.vision:
        raise ValueError("当前模型未登记视觉能力，请改用支持视觉的模型或关闭画面理解。")
    if tools and not value.tool_calling:
        raise ValueError("当前模型未登记工具调用能力，无法执行学习 Agent。")
