# 真实课程学习验收

目标是检验解释是否正确、是否回应学习者的困惑，以及追问后是否调整讲法。工具执行成功单独记录，不能据此判为教学通过。当前用脚本模拟学习者提问，由执行 Agent 对照资料复核；不冒充真人学习实验。

## 固定任务

`examples/suites/real_courses.json` 定义4组、12回合连续对话，问题和判断要点先于模型运行固定。题目引用公开课程，特意编写的错误速记用于检查多来源冲突，不是真实用户笔记。

- CS336：pre/post-norm 的图、恒等残差的反例、错误笔记核对。
- CS336：RMSNorm 的页面记法与常见定义、两维数值例子、FLOPs与运行时间图表。
- 中文课程：模型生成工具调用文字、外部执行器、结果回传、对工具结果的判断。
- 澄清与资料不足：缺少指代、澄清后的图表解释、不能从运行时间占比编造硬件加速测量。

评分维度为准确性、回应卡点、清晰度、追问适应（0–3）；无适用反馈时该维度可留空。严重事实错误、实际证据与回放相关性另列。脚本生成待复核模板，不自动给教学分或通过率。参考要点不会发送给讲解模型。

## 资料准备

媒体、讲义、字幕、原始回答和详细研究记录都留在本机 `data/`、`notes/`，不随代码分发。

| 本地文件（data/acceptance/real-courses/） | 课程出处 | 原视频范围 |
|---|---|---|
| cs336-architecture.pdf | [CS336 2025第三讲课程资料](https://cs336.stanford.edu/spring2025/) | 物理第10、14–16页 |
| cs336-prenorm.mp4 | [Stanford Online 第三讲](https://www.youtube.com/watch?v=ptFiH_bHnJw) | 353–551秒，片段198秒 |
| ntu-ai-agent.pdf | [李宏毅2025春课程](https://speech.ee.ntu.edu.tw/~hylee/ml/2025-spring.php) | 物理第49、50、61、62页 |
| ntu-tool-use.mp4 | [中文AI Agent第二讲](https://www.youtube.com/watch?v=M2Yg1kwPpts) | 3030–3300秒，片段270秒 |

PDF的精确下载地址保存在suite的`download_url`字段。视频原文件可以自行下载保存为 `cs336-original.mp4` / `ntu-original.mp4`，再本地截取：

```bash
ffmpeg -i data/acceptance/real-courses/cs336-original.mp4 -ss 353 -t 198 \
  -c:v libx264 -preset veryfast -crf 22 -c:a aac -movflags +faststart \
  data/acceptance/real-courses/cs336-prenorm.mp4
ffmpeg -i data/acceptance/real-courses/ntu-original.mp4 -ss 3030 -t 270 \
  -c:v libx264 -preset veryfast -crf 22 -c:a aac -movflags +faststart \
  data/acceptance/real-courses/ntu-tool-use.mp4
```

下载器报告成功不保证远程截片完整。已实际遇到视频容器时长正常、音轨提前结束的情况；准备器会核对片段与各轨道时长，不用缺失音频冒充ASR效果问题。原始文件可超过应用上传限制，上传的是截取后的短片。页码为PDF物理页，从1计数；不能直接把某些查看器的零基页号作为题目页码。

## 运行与复核

在 `.env` 配置讲解模型和本地ASR（`LA_ASR_PROVIDER=local`、`LA_ASR_MODEL=base`），安装可选ASR与开发依赖，系统需有ffmpeg/ffprobe。错误笔记由suite内容自动生成。

```bash
uv sync --extra dev --extra asr
# 可先准备资料；此步不调用讲解模型。等待结束后再运行评测，勿并发使用同一输出目录。
uv run python examples/evaluate_learning.py examples/suites/real_courses.json \
  --output data/evaluations/real-courses-v1 --prepare-only
# 会调用讲解API并产生费用
uv run python examples/evaluate_learning.py examples/suites/real_courses.json \
  --output data/evaluations/real-courses-v1
```

`--case cs_prenorm` 可选择任务组。相同输出目录复用已经准备的资料、跳过已记录的任务组；修改实现后做对比应使用新的输出目录，保留原结果。中断或失败的组保留其已完成回合，不将其自动计为完成，需要另起一次对照运行。

报告保存实际模型版本、ASR来源、原视频偏移、连续回答、工具观察、图像时间、回放链接检查、延迟及原始usage。用原视频起点加片段内时间换算原视频位置。位置检查只验证范围与曾查看的区间，是否包含关键解释仍需内容复核；此前回合已查看过的证据也可支持追问。

复核时先检查实质性错误，再判断是否回答当前卡点、是否清楚解释了理由。资料有歧义应保留原式并说明解释前提；不能默默改成模型熟悉的公式。没有对应测量时应说明不足，不能从其他图表占比推导出精确实测结论。自然语音ASR会误识别术语，需与PDF/画面或公开字幕核对；本轮不将课程字幕视为人工逐字真值来报告WER。

当前样本覆盖有限，连续提问也是人工设计。实际运行完成情况、失败与修复记录应以本地原始报告为依据，不宣传为对照实验已证明学习收益。

## 本轮运行记录

2026-10-19，使用配置的 `gpt-6.1-sol` 和本地CPU/int8 Whisper base。4组12回合基线执行完成；针对澄清入口另做3回合复测。执行 Agent 对照课件页图、实际工具观察和字幕核对了归一化结构、数值例子、表格与工具执行流程，未发现严重数学或图表事实错误。该判断不是独立人工评审，不能据此宣称用户理解率或教学通过率。

发现并处理的问题：

- 远程截片报告成功但内容或音轨不完整，改为下载原视频后本地截取，并在验收准备时核对各轨道时长。
- 澄清回答建议发送截图，但网页不支持独立截图附件。提示改为资料名、页码、媒体时间、原文或阅读器选中位置，复测不再要求该入口。
- 正文的页码/时间标记原先不可直接点击。现在依据当前及此前回合的实际观察解析可访问位置；文件名歧义、未查阅页面或区间仍保留为普通文字，不猜链接。检索片段可以支持对应位置，但不会被当作已经看过整页图像。

12回合基线的中位响应延迟为30.61秒；没有用推测单价换算费用。原始usage、回答、工具观察、复核理由保存在本机报告，详细错误排查记录放在忽略的 `notes/`。29项工程测试通过；内容复核与工程测试仍分开记录。

部分回答仍然偏长，自然语音中的专业术语也有误识别，需要结合PDF和画面核对。下一轮应着重长课程中的定位/检索，以及真实使用者对解释的反馈；本轮没有进行大样本实验或不同模型的对照排名。
