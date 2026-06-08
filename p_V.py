import os
import cv2
import argparse
from pathlib import Path


def natural_sort_key(path: Path):
    name = path.stem
    parts = []
    cur = ""
    is_digit = None

    for ch in name:
        if ch.isdigit():
            if is_digit is False:
                parts.append(cur)
                cur = ch
            else:
                cur += ch
            is_digit = True
        else:
            if is_digit is True:
                parts.append(int(cur))
                cur = ch
            else:
                cur += ch
            is_digit = False

    if cur:
        if is_digit:
            parts.append(int(cur))
        else:
            parts.append(cur)

    return parts


def images_to_video(
    image_dir: str,
    output_path: str,
    fps: float = 30.0,
    extensions=(".jpg", ".jpeg", ".png"),
):
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    image_files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    image_files = sorted(image_files, key=natural_sort_key)

    if not image_files:
        raise RuntimeError(f"目录中没有找到图片: {image_dir}")

    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        raise RuntimeError(f"无法读取第一张图片: {image_files[0]}")

    height, width = first_frame.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")

    print(f"[INFO] 共找到 {len(image_files)} 张图片")
    print(f"[INFO] 视频尺寸: {width} x {height}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] 输出文件: {output_path}")

    count = 0
    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] 跳过无法读取的图片: {img_path}")
            continue

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)
        count += 1

    writer.release()
    print(f"[DONE] 视频生成完成，写入帧数: {count}")


def build_argparser():
    parser = argparse.ArgumentParser(description="将图片序列合成为视频")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="图片文件夹路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出视频路径，例如 output.mp4"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="视频帧率，默认 30"
    )
    return parser


if __name__ == "__main__":
    images_to_video(
        image_dir="/home/sunddy/Programming/FoundationPose/video_data/output_teach/vis",
        output_path="/home/sunddy/Programming/FoundationPose/video_data/output_teach/vis_video.mp4",
        fps=30,
    )