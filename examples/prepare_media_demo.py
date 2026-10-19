"""Generate an authored, narrated lesson fixture on macOS, not a real lecture benchmark."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas

from learning_agent.media import command, probe


def main():
    root = Path("data/acceptance/media-demo")
    root.mkdir(parents=True, exist_ok=True)
    texts = [
        (
            "Let y equal w times x. At w equals three and x equals two, y is six. "
            "When x changes, keep w fixed. A small increase in x changes y by three times as much. "
            "So the derivative of y with respect to x is three, not two."
        ),
        (
            "Now let the loss be y squared. The derivative of the loss with respect to y is twice y, "
            "which is twelve here. The chain rule multiplies twelve by three, giving thirty six for x. "
            "For w, multiply twelve by two, giving twenty four. These are local sensitivities, not new values of x or w."
        ),
    ]
    titles = ["1. Which variable changes?", "2. Follow the chain"]
    lines = [
        [
            "y = w x",
            "w = 3     x = 2     y = 6",
            "Hold w fixed: dy/dx = w = 3",
            "Hold x fixed: dy/dw = x = 2",
        ],
        ["L = y^2        y = w x", "dL/dy = 2y = 12", "dL/dx = 12 x 3 = 36", "dL/dw = 12 x 2 = 24"],
    ]
    font = ImageFont.load_default(size=35)
    durations = []
    doc = canvas.Canvas(str(root / "companion.pdf"), pagesize=(700, 450))
    for i, (text, title, content) in enumerate(zip(texts, titles, lines, strict=True)):
        audio, picture, video = (root / f"scene-{i}.{suffix}" for suffix in ["aiff", "png", "mp4"])
        command(["say", "-v", "Samantha", "-r", "155", "-o", str(audio), text])
        durations.append(probe(audio)["duration"])
        image = Image.new("RGB", (1100, 650), "#f6f8f5")
        draw = ImageDraw.Draw(image)
        draw.text((70, 65), title, font=font, fill="#234734")
        for n, line in enumerate(content):
            draw.text((70, 180 + n * 85), line, font=font, fill="#253329")
        draw.rectangle((65, 550, 1035, 570), fill="#d78042" if i == 0 else "#4187b6")
        image.save(picture)
        command(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-loop",
                "1",
                "-i",
                str(picture),
                "-i",
                str(audio),
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "12",
                "-c:a",
                "aac",
                "-shortest",
                "-t",
                str(durations[-1]),
                str(video),
            ]
        )
        doc.setFont("Helvetica-Bold", 22)
        doc.drawString(45, 400, title)
        doc.setFont("Helvetica", 18)
        for n, line in enumerate(content):
            doc.drawString(45, 330 - n * 48, line)
        doc.showPage()
    doc.save()
    concat = root / "concat.txt"
    concat.write_text("\n".join(f"file '{(root / f'scene-{i}.mp4').resolve()}'" for i in range(2)))
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(root / "lesson.mp4"),
        ]
    )
    (root / "narration.txt").write_text("\n\n".join(texts))
    print(
        {
            "file": str(root / "lesson.mp4"),
            "scene_durations": durations,
            "duration": probe(root / "lesson.mp4")["duration"],
        }
    )


if __name__ == "__main__":
    main()
