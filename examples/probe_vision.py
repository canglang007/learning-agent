"""Paid opt-in image input probe; sends only a generated card, never the key."""

import asyncio
import json
import secrets
import string

from PIL import Image, ImageDraw, ImageFont

from learning_agent.config import Settings
from learning_agent.provider import ModelError, OpenAIProvider


async def main():
    settings = Settings()
    if not settings.api_key.get_secret_value():
        raise SystemExit("请先配置 LA_API_KEY。")
    root = settings.data_root / "probes"
    root.mkdir(parents=True, exist_ok=True)
    code = "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    image = Image.new("RGB", (800, 450), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)
    draw.rectangle((50, 40, 750, 160), fill="#dae9ff", outline="#174090", width=4)
    draw.text((100, 65), code, fill="black", font=font)
    for x in (160, 400, 640):
        draw.ellipse((x - 40, 260, x + 40, 340), fill="#e02030")
    path = root / "vision-card.png"
    image.save(path)
    try:
        message, usage = await OpenAIProvider(settings).step(
            [
                {
                    "role": "user",
                    "content": '只根据图片读取框里的字母数字，并数出红色圆形，返回 JSON：{"code":"...","red_circles":0}。',
                    "attachments": [{"asset_path": str(path.relative_to(settings.data_root))}],
                }
            ],
            "这是视觉输入能力检查。看不到图时明确说无法确认，不要猜。",
            [],
            False,
        )
        result = {
            "model": settings.model,
            "protocol": settings.protocol,
            "expected": {"code": code, "red_circles": 3},
            "answer": message["content"],
            "usage": usage,
        }
    except ModelError as error:
        result = {"state": "failed", "error": str(error)}
    (root / "vision-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
