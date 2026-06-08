#!/usr/bin/env python3
"""
使用本地 Ollama + Qwen 视觉模型，对机器人作业视频进行抽帧分析，
并把连续作业过程分解为原子动作。

推荐：使用视觉版 Qwen 模型，例如 qwen2.5vl:7b。
如果你本地装的是纯文本模型，它无法直接理解视频帧。

流程：
1. 读取视频元信息
2. 均匀抽取关键帧
3. 用 Ollama 多模态接口分析各帧事件
4. 再用一次文本整理，把事件合并成原子动作序列
5. 输出 JSON / TXT 结果
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import requests


# ============================================================
# 默认配置：已按你的环境写入程序
# 直接运行 python video_atomic_actions_ollama_qwen_qwen35b_default.py 即可
# ============================================================
DEFAULT_VIDEO_PATH = "/home/robot4/Programming/FoundationPose/video/5.22/color_video.mp4"
DEFAULT_MODEL_NAME = "qwen3.5:35B"
DEFAULT_OUTPUT_DIR = "/home/robot4/Programming/FoundationPose/video/5.22/qwen_atomic_actions_output"


@dataclass
class SampledFrame:
    sample_index: int
    frame_index: int
    timestamp_sec: float
    image_path: str


@dataclass
class AtomicAction:
    action_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    primary_actor: Optional[str]
    secondary_actor: Optional[str]
    object_name: Optional[str]
    action_name: str
    state_description: str
    precondition: str
    postcondition: str
    evidence: str


class OllamaError(RuntimeError):
    pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def video_info(video_path: str) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0.0 or frame_count <= 0:
        raise RuntimeError(f"视频元信息异常: fps={fps}, frame_count={frame_count}")

    return fps, frame_count, width, height


def build_uniform_sample_indices(frame_count: int, max_frames: int) -> List[int]:
    max_frames = max(1, min(max_frames, frame_count))
    if max_frames == 1:
        return [0]

    indices: List[int] = []
    for i in range(max_frames):
        idx = int(round(i * (frame_count - 1) / (max_frames - 1)))
        if not indices or idx != indices[-1]:
            indices.append(idx)
    return indices


def extract_sampled_frames(
    video_path: str,
    output_dir: Path,
    fps: float,
    sample_indices: Sequence[int],
    jpeg_quality: int = 92,
) -> List[SampledFrame]:
    ensure_dir(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    samples: List[SampledFrame] = []

    for sample_index, frame_index in enumerate(sample_indices):
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        if not ok:
            cap.release()
            raise RuntimeError(f"无法跳转到帧 {frame_index}")

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"读取帧失败: {frame_index}")

        image_path = output_dir / f"sample_{sample_index:03d}_frame_{frame_index:06d}.jpg"
        saved = cv2.imwrite(
            str(image_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not saved:
            cap.release()
            raise RuntimeError(f"保存抽帧失败: {image_path}")

        samples.append(
            SampledFrame(
                sample_index=sample_index,
                frame_index=int(frame_index),
                timestamp_sec=float(frame_index / fps),
                image_path=str(image_path),
            )
        )

    cap.release()
    return samples


def create_contact_sheet(samples: Sequence[SampledFrame], output_path: Path, thumb_w: int = 280) -> None:
    if not samples:
        return

    images = []
    for s in samples:
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_w / max(1, w)
        thumb_h = int(round(h * scale))
        img = cv2.resize(img, (thumb_w, thumb_h))
        label = f"#{s.sample_index}  f={s.frame_index}  t={s.timestamp_sec:.2f}s"
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        images.append(img)

    if not images:
        return

    cols = min(3, len(images))
    rows = math.ceil(len(images) / cols)
    cell_h = max(im.shape[0] for im in images)
    cell_w = max(im.shape[1] for im in images)
    canvas = 255 * np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        y = r * cell_h
        x = c * cell_w
        canvas[y : y + im.shape[0], x : x + im.shape[1]] = im

    cv2.imwrite(str(output_path), canvas)


# lazy import numpy for contact sheet only
import numpy as np


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout_s: int = 180,
    temperature: float = 0.1,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }

    resp = requests.post(url, json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise OllamaError(f"Ollama 请求失败: {resp.status_code} {resp.text[:500]}")

    data = resp.json()
    message = data.get("message", {})
    content = message.get("content", "")
    if not content:
        raise OllamaError("Ollama 返回为空")
    return str(content)


def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError("模型输出中没有可解析的 JSON")


def build_frame_analysis_prompt(samples: Sequence[SampledFrame]) -> str:
    frame_table = [
        {
            "sample_index": s.sample_index,
            "frame_index": s.frame_index,
            "timestamp_sec": round(s.timestamp_sec, 3),
        }
        for s in samples
    ]
    return (
        "你正在分析一个机器人作业视频的抽帧序列。\n"
        "请按图片顺序判断每一帧的作业阶段，并为每一帧补充状态描述。\n"
        "这里的状态描述要包括：机器人/手臂状态、目标物体状态、相对运动状态、是否接触/抓取/放下。\n"
        "不要长篇解释，只输出 JSON。\n"
        "JSON 格式固定为：\n"
        "{\n"
        '  "frames": [\n'
        "    {\n"
        '      "sample_index": 0,\n'
        '      "frame_index": 0,\n'
        '      "timestamp_sec": 0.0,\n'
        '      "primary_actor": "right_arm",\n'
        '      "secondary_actor": null,\n'
        '      "object_name": "metal_part",\n'
        '      "stage": "approach",\n'
        '      "robot_state": "右臂在目标物体上方靠近，末端尚未稳定接触",\n'
        '      "object_state": "物体静止放置在工作台/泡沫板上",\n'
        '      "contact_state": "未接触/接近/接触/稳定抓取/释放 中选择一个并简述",\n'
        '      "motion_state": "静止/轻微移动/抬起/平移/放下 中选择一个并简述",\n'
        '      "state_description": "一句完整状态描述，说明当前帧机器人、物体、接触和运动关系",\n'
        '      "observation": "一句简短视觉证据"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "stage 只允许使用这些枚举之一：idle, approach, contact, grasp, lift, transport, place, release, handover, support, adjust\n"
        "contact_state 建议使用：no_contact, approaching, contact, stable_grasp, dual_hand_contact, released\n"
        "motion_state 建议使用：static, small_motion, lifting, transporting, placing, stopped\n"
        f"帧元数据：{json.dumps(frame_table, ensure_ascii=False)}"
    )

def analyze_frames_with_qwen(
    base_url: str,
    model: str,
    samples: Sequence[SampledFrame],
    timeout_s: int,
) -> Dict[str, Any]:
    prompt = build_frame_analysis_prompt(samples)
    images_b64 = [image_to_base64(s.image_path) for s in samples]
    messages = [
        {
            "role": "user",
            "content": prompt,
            "images": images_b64,
        }
    ]
    raw = ollama_chat(base_url=base_url, model=model, messages=messages, timeout_s=timeout_s)
    return extract_json_block(raw)


def build_atomic_action_prompt(
    fps: float,
    frame_count: int,
    sampled_frames_json: Dict[str, Any],
) -> str:
    return (
        "你要根据机器人作业视频的抽帧观察结果，把整段视频的作业顺序分解成原子动作。\n"
        "请把相邻、连续、语义相同的帧级阶段合并成动作段。\n"
        "每一个原子动作都必须包含 state_description，用来描述该动作段内机器人/手臂、物体、接触和运动状态。\n"
        "输出 JSON，不要输出 Markdown。\n"
        "JSON 格式固定为：\n"
        "{\n"
        '  "atomic_actions": [\n'
        "    {\n"
        '      "action_id": 1,\n'
        '      "start_frame": 0,\n'
        '      "end_frame": 30,\n'
        '      "start_time_sec": 0.0,\n'
        '      "end_time_sec": 1.0,\n'
        '      "primary_actor": "right_arm",\n'
        '      "secondary_actor": null,\n'
        '      "object_name": "metal_part",\n'
        '      "action_name": "approach",\n'
        '      "state_description": "这一段内右臂从悬停位置接近目标物体，物体仍保持静止，尚未形成稳定抓取",\n'
        '      "precondition": "一句动作开始前状态",\n'
        '      "postcondition": "一句动作结束后状态",\n'
        '      "evidence": "一句视觉证据"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "action_name 只允许使用：approach, contact, grasp, lift, transport, place, release, handover, support, adjust, idle\n"
        "state_description 必须用中文描述状态，重点说明：谁在操作、物体是否静止/移动/抬起/放下、接触是否建立、是否双臂协同。\n"
        f"视频信息：fps={fps}, frame_count={frame_count}\n"
        f"抽帧观察结果：{json.dumps(sampled_frames_json, ensure_ascii=False)}"
    )

def build_summary_text(actions: Sequence[AtomicAction]) -> str:
    lines = []
    for a in actions:
        lines.append(
            f"{a.action_id:02d}. {a.action_name} | {a.primary_actor} | {a.object_name} | "
            f"frame {a.start_frame}-{a.end_frame} | {a.start_time_sec:.2f}-{a.end_time_sec:.2f}s"
        )
        lines.append(f"    state: {a.state_description}")
        lines.append(f"    precondition: {a.precondition}")
        lines.append(f"    postcondition: {a.postcondition}")
        lines.append(f"    evidence: {a.evidence}")
    return "\n".join(lines)


def parse_atomic_actions(data: Dict[str, Any]) -> List[AtomicAction]:
    actions_raw = data.get("atomic_actions", [])
    actions: List[AtomicAction] = []
    for item in actions_raw:
        actions.append(
            AtomicAction(
                action_id=int(item.get("action_id", len(actions) + 1)),
                start_frame=int(item["start_frame"]),
                end_frame=int(item["end_frame"]),
                start_time_sec=float(item["start_time_sec"]),
                end_time_sec=float(item["end_time_sec"]),
                primary_actor=item.get("primary_actor"),
                secondary_actor=item.get("secondary_actor"),
                object_name=item.get("object_name"),
                action_name=str(item.get("action_name", "unknown")),
                state_description=str(item.get("state_description", item.get("state", ""))),
                precondition=str(item.get("precondition", "")),
                postcondition=str(item.get("postcondition", "")),
                evidence=str(item.get("evidence", "")),
            )
        )
    return actions


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    frames_dir = out_dir / "sampled_frames"
    ensure_dir(out_dir)
    ensure_dir(frames_dir)

    print("=" * 80)
    print("[DEFAULT] video:", args.video)
    print("[DEFAULT] model:", args.model)
    print("[DEFAULT] output_dir:", args.output_dir)
    print("=" * 80)

    fps, frame_count, width, height = video_info(args.video)
    duration_sec = frame_count / fps

    print("=" * 80)
    print("[INFO] 视频信息")
    print(f"video       : {args.video}")
    print(f"fps         : {fps:.3f}")
    print(f"frame_count : {frame_count}")
    print(f"resolution  : {width}x{height}")
    print(f"duration_s  : {duration_sec:.3f}")
    print("=" * 80)

    sample_indices = build_uniform_sample_indices(frame_count, args.max_frames)
    samples = extract_sampled_frames(
        video_path=args.video,
        output_dir=frames_dir,
        fps=fps,
        sample_indices=sample_indices,
        jpeg_quality=args.jpeg_quality,
    )

    print(f"[INFO] 已抽帧 {len(samples)} 张")
    for s in samples:
        print(f"  sample#{s.sample_index:02d} frame={s.frame_index:06d} t={s.timestamp_sec:.2f}s")

    contact_sheet_path = out_dir / "contact_sheet.jpg"
    create_contact_sheet(samples, contact_sheet_path)

    print("[QWEN] 正在分析抽帧，并生成每帧状态描述...")
    frame_analysis = analyze_frames_with_qwen(
        base_url=args.ollama_base_url,
        model=args.model,
        samples=samples,
        timeout_s=args.timeout_s,
    )
    save_json(out_dir / "frame_analysis.json", frame_analysis)

    atomic_prompt = build_atomic_action_prompt(
        fps=fps,
        frame_count=frame_count,
        sampled_frames_json=frame_analysis,
    )
    print("[QWEN] 正在合并原子动作，并生成每个动作段的状态描述...")
    atomic_raw = ollama_chat(
        base_url=args.ollama_base_url,
        model=args.model,
        messages=[{"role": "user", "content": atomic_prompt}],
        timeout_s=args.timeout_s,
    )
    atomic_json = extract_json_block(atomic_raw)
    actions = parse_atomic_actions(atomic_json)

    atomic_payload = {
        "video": str(Path(args.video).resolve()),
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "model": args.model,
        "atomic_actions": [asdict(a) for a in actions],
    }
    save_json(out_dir / "atomic_actions.json", atomic_payload)
    save_text(out_dir / "atomic_actions.txt", build_summary_text(actions))

    print("=" * 80)
    print("[DONE] 原子动作分解完成")
    print(f"抽帧目录       : {frames_dir}")
    print(f"拼图预览       : {contact_sheet_path}")
    print(f"帧级分析       : {out_dir / 'frame_analysis.json'}")
    print(f"原子动作 JSON  : {out_dir / 'atomic_actions.json'}")
    print(f"原子动作 TXT   : {out_dir / 'atomic_actions.txt'}")
    print("=" * 80)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用本地 Ollama + Qwen 视觉模型分析视频，并分解成原子动作"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=DEFAULT_VIDEO_PATH,
        help="输入视频路径。默认写死为 5.11/color_video.mp4",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="本地 Ollama 模型名。当前默认写死为 qwen3.5:35B。",
    )
    parser.add_argument(
        "--ollama_base_url",
        type=str,
        default="http://127.0.0.1:11434",
        help="Ollama 服务地址",
    )
    parser.add_argument("--max_frames", type=int, default=10, help="最多抽取多少张关键帧。35B 模型较慢，默认 8 张")
    parser.add_argument("--timeout_s", type=int, default=600, help="每次模型请求超时秒数。35B 模型较慢，默认 600 秒")
    parser.add_argument("--jpeg_quality", type=int, default=92)
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)