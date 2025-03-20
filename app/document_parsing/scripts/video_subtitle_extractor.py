#!/usr/bin/env python
"""
简化版字幕提取工具
直接使用faster_whisper库提取字幕
"""

import json
import os
import warnings
from datetime import timedelta
from rich.progress import track
from faster_whisper import WhisperModel


def format_timestamp(seconds: float) -> str:
    """格式化时间戳为SRT格式"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def extract_subtitle(file_path: str, output_format: str = "srt", language: str | None = None) -> str:
    """
    从媒体文件提取字幕

    Args:
        file_path: 媒体文件路径
        output_format: 输出格式 (srt, txt, json)
        language: 语言代码 (如 "zh", "en")

    Returns:
        str: 输出文件路径
    """
    try:
        warnings.filterwarnings("ignore")

        print("加载模型中...")
        model = WhisperModel("base", device="auto")

        print(f"处理文件: {file_path}")
        segments, info = model.transcribe(file_path, language=language)

        subtitle_dir = os.path.join(os.path.dirname(file_path), "subtitle")
        os.makedirs(subtitle_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(subtitle_dir, f"{base_name}.{output_format}")

        if output_format == "srt":
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in track(enumerate(segments, 1), desc="生成SRT字幕"):
                    start = format_timestamp(segment.start)
                    end = format_timestamp(segment.end)
                    text = segment.text.strip()

                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")

        elif output_format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for segment in track(segments, desc="生成TXT字幕"):
                    f.write(f"{segment.text.strip()}\n")

        elif output_format == "json":
            segments_list = []
            for segment in track(segments, desc="生成JSON字幕"):
                segments_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(segments_list, f, ensure_ascii=False, indent=2)

        else:
            raise ValueError(f"不支持的输出格式: {output_format}")

        print(f"字幕已保存到: {output_path}")
        return output_path

    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise


if __name__ == "__main__":
    video_path = r"C:\Users\STARTER\Desktop\test.mp4"
    output_path = extract_subtitle(video_path, "srt", "zh")
