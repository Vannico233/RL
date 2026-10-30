#!/usr/bin/env python3
"""Build an offline HTML case viewer with original-coordinate crop overlays."""

import argparse
from html import escape
import json
import os
from pathlib import Path


def render(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    state = json.loads((run / "run.json").read_text(encoding="utf-8"))
    cards = []
    for path in sorted((run / "cases").glob("*/trajectory.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        observation, prediction = trace["observation"], trace["prediction"]
        width, height = observation["width"], observation["height"]
        source = escape(os.path.relpath(path.parent / trace["original_image"], output.parent).replace("\\", "/"), quote=True)
        overlay = []
        for index, event in enumerate(prediction["tool_events"]):
            if event["success"]:
                x1, y1, x2, y2 = event["executed_bbox_original"]
                color = "#d9462c" if index == 0 else "#1673c9"
                overlay.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{color}" stroke-width="{max(width/250, 2)}"/>')
                overlay.append(f'<text x="{x1+3}" y="{max(y1+18,18)}" fill="{color}" font-size="{max(width/35,14)}">Crop {index+1}</text>')
        svg = f'<svg viewBox="0 0 {width} {height}" role="img"><image href="{source}" width="{width}" height="{height}"/>{"".join(overlay)}</svg>'
        history = []
        for message in trace["messages"]:
            content = message["content"]
            blocks = []
            if isinstance(content, str):
                blocks.append("<pre>" + escape(content) + "</pre>")
            else:
                for block in content:
                    if block["type"] == "image":
                        src = escape(os.path.relpath(path.parent / block["path"], output.parent).replace("\\", "/"), quote=True)
                        blocks.append(f'<img src="{src}" alt="Actual model observation" loading="lazy"/>')
                    else:
                        blocks.append("<pre>" + escape(block["text"]) + "</pre>")
            history.append(f'<div class="message"><b>{escape(message["role"])}</b>{"".join(blocks)}</div>')
        result = "正确" if trace["score"]["relaxed"] else "错误 / 解析失败"
        cards.append(f'<article><h2>{escape(observation["question"])}</h2><p>{result} · 策略 {escape(prediction["policy"])}</p>'
                     f'{svg}<p>最终输出：{escape(prediction["completion"])}</p>'
                     f'<details><summary>完整交互与真实观察图片</summary>{"".join(history)}</details>'
                     f'<details><summary>成本与动作记录</summary><pre>{escape(json.dumps(prediction, ensure_ascii=False, indent=2))}</pre></details></article>')
    if not cards:
        raise ValueError("No saved cases found; evaluate with a positive --case-limit first")
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ChartQA 交互案例</title>'
    html += '<style>body{font:16px/1.6 system-ui,sans-serif;margin:30px auto;max-width:1100px;padding:0 20px;background:#f5f6f8;color:#20252b}article{background:white;padding:24px;margin:24px 0;border-radius:12px}svg,img{max-width:100%;max-height:650px}svg{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f3f5;padding:12px}summary{cursor:pointer;font-weight:600}.message{border-left:3px solid #8ba5be;margin:16px 0;padding-left:16px}</style>'
    html += f'<h1>图表问答交互案例</h1><p>运行状态：{escape(state["status"])}。案例按预先声明的成功/失败配额保存；总体结果以完整题集评分为准。</p>'
    html += "".join(cards) + "</html>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return len(cards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    target = args.output or str(Path(args.run) / "cases/index.html")
    print(f"Rendered {render(args.run, target)} cases to {target}")
