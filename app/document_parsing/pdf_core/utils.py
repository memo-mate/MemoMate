#!/usr/bin/env python

import os
import re
from typing import Any

import pikepdf

from app.core import logger

# 检查是否安装了pikepdf
HAS_PIKEPDF = True
try:
    import pikepdf
except ImportError:
    HAS_PIKEPDF = False
    logger.warning("未安装pikepdf库，某些功能将受限")


def get_default_output_path(input_path: str, suffix: str = "_processed") -> str:
    """获取默认输出文件路径

    Args:
        input_path: 输入文件路径
        suffix: 输出文件后缀

    Returns:
        默认输出文件路径
    """
    base_name, ext = os.path.splitext(input_path)
    return f"{base_name}{suffix}{ext}"


class PDFUtils:
    """PDF处理工具类"""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """从PDF提取文本内容

        Args:
            pdf_path: PDF文件路径

        Returns:
            提取的文本内容
        """
        try:
            # 使用pdfminer.six提取文本，设置编码为utf-8
            from pdfminer.high_level import extract_text
            from pdfminer.layout import LAParams

            # 设置LAParams以更好地处理文本布局
            laparams = LAParams(line_overlap=0.5, char_margin=2.0, line_margin=0.5, word_margin=0.1, all_texts=True)

            # 提取文本，指定编码
            text = extract_text(pdf_path, laparams=laparams, codec="utf-8")

            return text
        except Exception as e:
            logger.error(f"提取文本时出错: {e}")
            return ""

    @staticmethod
    def extract_images_from_pdf(pdf_path: str, output_dir: str) -> list[str]:
        """从PDF提取图片

        Args:
            pdf_path: PDF文件路径
            output_dir: 图片输出目录

        Returns:
            提取的图片路径列表
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_paths = []

        try:
            with pikepdf.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    for j, (_name, raw_image) in enumerate(page.images.items()):
                        image = pikepdf.PdfImage(raw_image)
                        output_path = os.path.join(output_dir, f"page{i + 1}_img{j + 1}.png")

                        with open(output_path, "wb") as f:
                            f.write(image.read_bytes())

                        image_paths.append(output_path)

            return image_paths
        except Exception as e:
            logger.error(f"提取图片时出错: {e}")
            return []

    @staticmethod
    def detect_watermark_type(pdf_path: str) -> dict[str, Any]:
        """检测PDF中的水印类型

        Args:
            pdf_path: PDF文件路径

        Returns:
            水印类型信息的字典
        """
        logger.info(f"开始检测文件 {pdf_path} 中的水印")

        result: dict[str, Any] = {
            "has_watermark": False,
            "watermark_types": [],
            "text_watermarks": [],
            "image_watermarks": [],
            "watermark_details": {
                "text_occurrences": {},  # 文本水印出现次数
                "text_confidence": {},  # 文本水印置信度
                "image_features": [],  # 图像水印特征
            },
        }

        # 提取文本检查文本水印
        logger.info("提取PDF文本内容进行水印检测...")
        text = PDFUtils.extract_text_from_pdf(pdf_path)

        # 常见水印文本模式
        watermark_patterns = [
            r"机密|保密|内部文件|CONFIDENTIAL|DRAFT|草稿",
            r"版权所有|COPYRIGHT|ALL RIGHTS RESERVED",
            r"禁止复制|DO NOT COPY|COPY PROHIBITED",
            r"未经授权|UNAUTHORIZED|SAMPLE|样本",
            r"DEMO|演示|测试|TEST",
        ]

        # 直接检查特定关键词（不使用正则表达式）
        keywords = [
            "机密",
            "保密",
            "内部文件",
            "CONFIDENTIAL",
            "DRAFT",
            "草稿",
            "版权所有",
            "COPYRIGHT",
            "ALL RIGHTS RESERVED",
            "禁止复制",
            "DO NOT COPY",
            "COPY PROHIBITED",
            "未经授权",
            "UNAUTHORIZED",
            "SAMPLE",
            "样本",
            "DEMO",
            "演示",
            "测试",
            "TEST",
        ]

        # 先使用关键词直接检查
        for keyword in keywords:
            if keyword in text:
                # 计算关键词出现次数
                occurrences = text.count(keyword)
                result["watermark_details"]["text_occurrences"][keyword] = occurrences

                # 根据出现次数估计置信度
                confidence = min(0.5 + (occurrences / 10) * 0.5, 0.95)  # 最高置信度0.95
                result["watermark_details"]["text_confidence"][keyword] = confidence

                result["has_watermark"] = True
                if "text" not in result["watermark_types"]:
                    result["watermark_types"].append("text")
                result["text_watermarks"].append(keyword)
                logger.info(f"检测到关键词水印: {keyword}, 出现次数: {occurrences}, 置信度: {confidence:.2f}")

        # 再使用正则表达式进行更复杂的匹配
        for pattern in watermark_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    if match not in result["text_watermarks"]:
                        result["has_watermark"] = True
                        if "text" not in result["watermark_types"]:
                            result["watermark_types"].append("text")
                        result["text_watermarks"].append(match)

                        # 估计置信度 (正则匹配的置信度稍低)
                        occurrences = text.count(match)
                        result["watermark_details"]["text_occurrences"][match] = occurrences
                        confidence = min(0.4 + (occurrences / 10) * 0.5, 0.9)
                        result["watermark_details"]["text_confidence"][match] = confidence
                        logger.info(
                            f"通过正则表达式检测到水印: {match}, 出现次数: {occurrences}, 置信度: {confidence:.2f}"
                        )

        # 检查图片水印
        logger.info("检查PDF中的图片水印...")
        try:
            with pikepdf.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    if "/XObject" in page.keys():
                        for key, obj in page["/XObject"].items():
                            if obj.get("/Subtype") == "/Image":
                                image_info = {"page": page_idx + 1, "key": str(key), "features": []}
                                is_watermark = False
                                confidence = 0.0

                                # 检查图片是否可能是水印
                                if obj.get("/SMask") is not None:
                                    is_watermark = True
                                    image_info["features"].append("有SMask (半透明)")
                                    confidence += 0.4

                                if obj.get("/Mask") is not None:
                                    is_watermark = True
                                    image_info["features"].append("有Mask")
                                    confidence += 0.3

                                # 检查图像尺寸
                                if "/Width" in obj and "/Height" in obj:
                                    width = int(obj["/Width"])
                                    height = int(obj["/Height"])
                                    image_info["size"] = (width, height)

                                    # 检查页面尺寸
                                    if "/MediaBox" in page:
                                        media_box = page["/MediaBox"]
                                        page_width = float(media_box[2]) - float(media_box[0])
                                        page_height = float(media_box[3]) - float(media_box[1])

                                        # 如果图像尺寸接近页面尺寸，可能是水印
                                        width_ratio = width / page_width
                                        height_ratio = height / page_height

                                        if width_ratio > 0.9 and height_ratio > 0.9:
                                            is_watermark = True
                                            image_info["features"].append(
                                                f"覆盖整页 ({width_ratio:.2f}x{height_ratio:.2f})"
                                            )
                                            confidence += 0.3

                                # 检查颜色空间
                                if "/ColorSpace" in obj:
                                    color_space = obj["/ColorSpace"]
                                    image_info["color_space"] = str(color_space)

                                    # 如果是灰度图像，更可能是水印
                                    if str(color_space) == "/DeviceGray":
                                        image_info["features"].append("灰度图像")
                                        confidence += 0.2

                                if is_watermark:
                                    image_info["confidence"] = min(confidence, 0.95)
                                    result["has_watermark"] = True
                                    if "image" not in result["watermark_types"]:
                                        result["watermark_types"].append("image")
                                    result["image_watermarks"].append(image_info)
                                    result["watermark_details"]["image_features"].append(image_info)
                                    logger.info(
                                        f"检测到图片水印: 页面 {page_idx + 1}, 特征: {image_info['features']}, 置信度: {image_info['confidence']:.2f}"
                                    )
        except Exception as e:
            logger.error(f"检测图片水印时出错: {e}")

        # 总结检测结果
        if result["has_watermark"]:
            watermark_types = ", ".join(result["watermark_types"])
            logger.info(f"检测完成，发现水印类型: {watermark_types}")
            if "text" in result["watermark_types"]:
                logger.info(f"文本水印: {', '.join(result['text_watermarks'])}")
            if "image" in result["watermark_types"]:
                logger.info(f"图片水印数量: {len(result['image_watermarks'])}")
        else:
            logger.info("未检测到水印")

        return result

    @staticmethod
    def get_pdf_info(pdf_path: str) -> dict[str, Any]:
        """获取PDF文件信息

        Args:
            pdf_path: PDF文件路径

        Returns:
            PDF信息的字典
        """
        info: dict[str, Any] = {"page_count": 0, "metadata": {}, "has_watermark": False, "watermark_types": []}

        try:
            with pikepdf.open(pdf_path) as pdf:
                info["page_count"] = len(pdf.pages)

                if pdf.docinfo:
                    for key, value in pdf.docinfo.items():
                        info["metadata"][str(key)[1:]] = str(value)

            # 检测水印
            watermark_info = PDFUtils.detect_watermark_type(pdf_path)
            info["has_watermark"] = watermark_info["has_watermark"]
            info["watermark_types"] = watermark_info["watermark_types"]
            info["watermark_details"] = watermark_info.get("watermark_details", {})

            return info
        except Exception as e:
            logger.error(f"获取PDF信息时出错: {e}")
            return info
