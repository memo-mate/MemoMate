import os
import re
import shutil
import tempfile
from typing import Any

import numpy as np
import pikepdf
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTTextBoxHorizontal, LTTextLineHorizontal
from pikepdf import Name, Object, PdfImage

from app.core import logger
from app.document_parsing.pdf_core.utils import PDFUtils


class WatermarkRemover:
    """PDF水印去除器"""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.utils = PDFUtils()

        # 临时文件和目录列表，用于跟踪和清理
        self._temp_files = []
        self._temp_dirs = [self.temp_dir]

        # 常见水印关键词
        self.common_watermarks = [
            "机密",
            "保密",
            "内部",
            "草稿",
            "DRAFT",
            "CONFIDENTIAL",
            "INTERNAL",
            "请勿外传",
            "版权所有",
            "COPYRIGHT",
            "DO NOT COPY",
            "禁止复制",
            "未经授权",
            "UNAUTHORIZED",
            "SAMPLE",
            "样本",
            "DEMO",
            "演示",
            "测试",
            "TEST",
        ]

    def __del__(self):
        """析构函数，确保临时资源被清理"""
        self.cleanup()

    def cleanup(self):
        """清理所有临时资源"""
        # 清理临时文件
        for file_path in self._temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.warning(f"清理临时文件 {file_path} 失败: {e}")

        # 清理临时目录
        for dir_path in self._temp_dirs:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
            except Exception as e:
                logger.warning(f"清理临时目录 {dir_path} 失败: {e}")

        # 重置临时资源列表
        self._temp_files = []
        self._temp_dirs = [self.temp_dir]

    def _detect_text_watermarks(self, pdf_path: str) -> dict:
        """检测PDF中的文本水印

        Args:
            pdf_path: PDF文件路径

        Returns:
            文本水印信息字典，包含水印文本、出现次数、位置、角度等详细信息
        """
        text_watermarks_info = {
            "text_watermarks": [],  # 检测到的水印文本列表
            "text_frequency": {},  # 水印文本出现频率
            "watermarks_info": [],  # 水印实例详细信息
            "watermark_by_page": {},  # 按页面组织的水印信息
        }

        try:
            page_num = 0
            for page_layout in extract_pages(pdf_path):
                page_num += 1
                page_watermarks = []

                for element in page_layout:
                    if isinstance(element, LTTextBoxHorizontal):
                        for text_line in element:
                            if isinstance(text_line, LTTextLineHorizontal):
                                text = text_line.get_text().strip()

                                if not text:
                                    continue

                                # 检查此文本元素是否为水印
                                is_watermark, confidence, watermark_features = self.is_text_watermark(
                                    text_line, text, page_layout.width, page_layout.height
                                )

                                if not is_watermark:
                                    continue

                                # 添加到水印文本列表(如果不存在)
                                if text not in text_watermarks_info["text_watermarks"]:
                                    text_watermarks_info["text_watermarks"].append(text)

                                # 更新文本频率
                                if text in text_watermarks_info["text_frequency"]:
                                    text_watermarks_info["text_frequency"][text] += 1
                                else:
                                    text_watermarks_info["text_frequency"][text] = 1

                                # 记录文本位置和几何信息
                                bbox = text_line.bbox
                                x_center = (bbox[0] + bbox[2]) / 2
                                y_center = (bbox[1] + bbox[3]) / 2
                                width = bbox[2] - bbox[0]
                                height = bbox[3] - bbox[1]
                                rel_x = x_center / page_layout.width
                                rel_y = y_center / page_layout.height
                                rel_width = width / page_layout.width
                                rel_height = height / page_layout.height

                                # 收集颜色和透明度信息
                                opacity = 1.0  # 默认不透明
                                color_info = {"type": "unknown", "value": None}

                                # 从文本字符中提取颜色信息(如果可用)
                                for char in text_line:
                                    if isinstance(char, LTChar) and hasattr(char, "graphicstate"):
                                        if hasattr(char.graphicstate, "ncolor"):
                                            color = char.graphicstate.ncolor
                                            if len(color) >= 3:  # RGB颜色
                                                color_info = {"type": "rgb", "value": color}
                                                break

                                        # 尝试获取透明度信息
                                        if hasattr(char.graphicstate, "ca"):
                                            opacity = float(char.graphicstate.ca)
                                            break

                                # 记录水印位置信息
                                watermark_info = {
                                    "text": text,
                                    "page": page_num,
                                    "confidence": confidence,
                                    "rotation": watermark_features.get("rotation", 0),
                                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                                    "center": [x_center, y_center],
                                    "dimensions": [width, height],
                                    "rel_position": [rel_x, rel_y],
                                    "rel_dimensions": [rel_width, rel_height],
                                    "opacity": opacity,
                                    "color": color_info,
                                    "features": watermark_features,
                                }

                                # 添加到水印信息列表
                                text_watermarks_info["watermarks_info"].append(watermark_info)
                                page_watermarks.append(watermark_info)

                # 按页面保存水印信息
                if page_watermarks:
                    text_watermarks_info["watermark_by_page"][page_num] = page_watermarks
                    logger.info(f"第 {page_num} 页检测到 {len(page_watermarks)} 个文本水印")

                    for wm_idx, wm in enumerate(page_watermarks):
                        logger.debug(
                            f"  水印 {wm_idx + 1}: '{wm['text']}', "
                            f"角度: {wm['rotation']:.1f}°, "
                            f"位置: ({wm['rel_position'][0]:.2f}, {wm['rel_position'][1]:.2f}), "
                            f"透明度: {wm['opacity']:.2f}"
                        )

            # 记录水印统计信息
            if text_watermarks_info["watermarks_info"]:
                # 按角度分组的水印
                rotations = {}
                for wm in text_watermarks_info["watermarks_info"]:
                    rot = round(wm["rotation"])
                    if rot in rotations:
                        rotations[rot].append(wm)
                    else:
                        rotations[rot] = [wm]

                text_watermarks_info["rotation_groups"] = {str(rot): len(wms) for rot, wms in rotations.items()}

                # 记录主要角度
                if rotations:
                    main_rotation = max(rotations.items(), key=lambda x: len(x[1]))[0]
                    text_watermarks_info["main_rotation"] = main_rotation
                    logger.info(f"检测到的主要水印角度: {main_rotation}°")

        except Exception as e:
            logger.error(f"检测文本水印时出错: {e}")

        return text_watermarks_info

    def _analyze_pdf_structure(
        self, pdf_path: str, text_watermarks: list[str]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """分析PDF结构以识别可能包含水印的内容对象和XObject

        Args:
            pdf_path: PDF文件路径
            text_watermarks: 已识别的文本水印列表

        Returns:
            (内容对象信息列表, XObject信息列表)
        """
        content_objects_info = []
        xobject_info = []

        try:
            with pikepdf.open(pdf_path) as pdf:
                # 遍历每一页
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1

                    # 分析内容对象
                    if "/Contents" in page:
                        contents = page["/Contents"]

                        # 如果内容是数组
                        if isinstance(contents, list) and len(contents) > 1:
                            page_content_info = {
                                "page": page_num,
                                "content_count": len(contents),
                                "watermark_indices": [],
                            }

                            # 分析每个内容对象
                            for i, content_obj in enumerate(contents):
                                try:
                                    stream = content_obj.read_bytes()

                                    # 尝试多种编码解码内容
                                    stream_texts = []
                                    for encoding in ["utf-8", "latin-1", "utf-16be", "gbk"]:
                                        try:
                                            decoded = stream.decode(encoding, errors="ignore")
                                            stream_texts.append(decoded)
                                        except Exception:
                                            pass

                                    # 合并所有成功解码的文本
                                    stream_text = " ".join(stream_texts)

                                    # 检查是否包含水印文本
                                    is_watermark = False
                                    for text in text_watermarks:
                                        if text in stream_text:
                                            is_watermark = True
                                            page_content_info["watermark_indices"].append(i)
                                            logger.info(f"在第{page_num}页内容对象{i}中找到水印文本: {text}")
                                            break

                                    # 检查是否包含常见水印特征
                                    if not is_watermark and i > 0:  # 不检查第一个对象，可能是主要内容
                                        watermark_patterns = [
                                            rb"Tj\s+ET",  # 文本渲染操作符
                                            rb"0\s+Tc",  # 字符间距设置
                                            rb"0\s+Tw",  # 单词间距设置
                                            rb"BT\s+/F\d+",  # 文本块开始和字体选择
                                        ]

                                        for pattern in watermark_patterns:
                                            if re.search(pattern, stream):
                                                is_watermark = True
                                                if i not in page_content_info["watermark_indices"]:
                                                    page_content_info["watermark_indices"].append(i)
                                                logger.info(f"在第{page_num}页内容对象{i}中找到水印特征模式")
                                                break
                                except Exception as e:
                                    logger.warning(f"分析内容对象{i}时出错: {e}")

                            # 如果找到水印对象，添加到结果中
                            if page_content_info["watermark_indices"]:
                                content_objects_info.append(page_content_info)

                    # 分析XObject
                    if "/XObject" in page["/Resources"]:
                        resources = page["/Resources"]
                        xobjects = resources["/XObject"]

                        page_xobject_info = {"page": page_num, "xobject_keys": []}

                        # 检查每个XObject
                        for key in list(xobjects.keys()):
                            xobject = xobjects[key]
                            if xobject.get("/Subtype") == "/Form":  # 表单XObject可能包含水印
                                page_xobject_info["xobject_keys"].append(str(key))
                            elif xobject.get("/Subtype") == "/Image":  # 图像XObject也可能是水印
                                # 记录最后一个图像对象（通常是水印）
                                page_xobject_info["last_image_key"] = str(key)

                        if page_xobject_info["xobject_keys"] or "last_image_key" in page_xobject_info:
                            xobject_info.append(page_xobject_info)

        except Exception as e:
            logger.error(f"分析PDF结构时出错: {e}")

        return content_objects_info, xobject_info

    def detect_watermark(self, pdf_path: str) -> dict[str, Any]:
        """检测PDF中的水印类型和特征

        Args:
            pdf_path: PDF文件路径

        Returns:
            水印类型信息的字典
        """
        logger.info(f"开始检测PDF水印: {pdf_path}")

        # 初始化结果字典
        result = {
            "has_watermark": False,
            "watermark_types": [],
            "text_watermarks_info": {},
            "image_watermarks_info": [],
            "watermark_details": {
                "text_confidence": {},
                "image_confidence": {},
                "content_objects_info": [],
                "xobject_info": [],
            },
        }

        try:
            # 检测文本水印
            text_result = self._detect_text_watermarks(pdf_path)

            # 如果检测到文本水印
            if text_result.get("watermarks_info", []):
                result["has_watermark"] = True
                if "text" not in result["watermark_types"]:
                    result["watermark_types"].append("text")

                # 保存文本水印信息
                result["text_watermarks_info"] = text_result

                # 提取置信度信息
                for wm in text_result.get("watermarks_info", []):
                    text = wm.get("text", "")
                    confidence = wm.get("confidence", 0)
                    if text:
                        result["watermark_details"]["text_confidence"][text] = confidence

                # 记录水印文本频率
                text_frequency = text_result.get("text_frequency", {})

                # 日志记录
                logger.info(f"检测到 {len(text_frequency.keys())} 种文本水印:")
                for key, value in text_frequency.items():
                    logger.info(f"水印 [{key}] 共出现 {value} 次")

            # 检测图片水印
            image_watermarks = self._detect_image_watermarks(pdf_path)

            if image_watermarks:
                result["has_watermark"] = True
                if "image" not in result["watermark_types"]:
                    result["watermark_types"].append("image")
                result["watermark_details"]["image_watermarks"] = image_watermarks

            logger.info(f"水印检测完成: {result['has_watermark']}")
            logger.info(f"发现的水印类型: {result['watermark_types']}")

            return result

        except Exception as e:
            logger.error(f"检测水印时出错: {e}")
            return result

    def _detect_image_watermarks(self, pdf_path: str) -> list[dict[str, Any]]:
        """检测PDF中的图片水印

        Args:
            pdf_path: PDF文件路径

        Returns:
            图片水印信息列表
        """
        image_watermarks = []

        try:
            with pikepdf.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1

                    # 处理页面上的图像
                    if "/XObject" in page["/Resources"]:
                        xobjects = page["/Resources"]["/XObject"]
                        for key, xobject in xobjects.items():
                            if xobject.get("/Subtype") == "/Image":
                                # 检查是否可能是水印图像
                                is_watermark, confidence, features = self.is_image_watermark(xobject, page)

                                if is_watermark:
                                    image_watermarks.append(
                                        {
                                            "page": page_num,
                                            "image_key": str(key),
                                            "confidence": confidence,
                                            "features": features,
                                        }
                                    )
                                    logger.info(f"在第 {page_num} 页检测到图片水印, 置信度: {confidence:.2f}")

        except Exception as e:
            logger.error(f"检测图片水印时出错: {e}")

        return image_watermarks

    def is_text_watermark(
        self, text_element: LTTextLineHorizontal, text: str, page_width: float, page_height: float
    ) -> tuple[bool, float, dict[str, Any]]:
        """判断文本元素是否为水印

        Args:
            text_element: 文本元素
            text: 文本内容
            page_width: 页面宽度
            page_height: 页面高度

        Returns:
            (是否为水印, 置信度, 特征字典)
        """
        confidence = 0.0
        features = {
            "keyword_match": False,  # 是否匹配常见水印关键字
            "central_position": False,  # 是否在页面中央
            "rotation": 0,  # 旋转角度
            "light_color": False,  # 是否为浅色
            "opacity": 1.0,  # 透明度
            "color": None,  # 颜色信息
            "font_size": 0,  # 字体大小
            "font_name": None,  # 字体名称
            "repeated_pattern": False,  # 是否为重复模式
        }

        # 特征1: 检查常见水印关键字
        contains_keyword = any(keyword in text for keyword in self.common_watermarks)
        if contains_keyword:
            confidence += 0.3
            features["keyword_match"] = True
            logger.debug(f"文本 '{text}' 匹配水印关键字")

        # 特征2: 检查文本位置（水印通常在页面中央或均匀分布）
        bbox = text_element.bbox
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        rel_x = x_center / page_width
        rel_y = y_center / page_height

        if 0.3 <= rel_x <= 0.7 and 0.3 <= rel_y <= 0.7:
            confidence += 0.1
            features["central_position"] = True
            logger.debug(f"文本 '{text}' 位于页面中央区域")

        # 特征3: 检查文本旋转
        char_rotations = []
        font_sizes = []
        font_names = set()
        opacity_values = []

        for char in text_element:
            if isinstance(char, LTChar):
                # 提取字体信息
                if hasattr(char, "fontname"):
                    font_names.add(char.fontname)
                if hasattr(char, "size"):
                    font_sizes.append(char.size)

                # 提取旋转信息
                if hasattr(char, "matrix"):
                    matrix = char.matrix
                    char_rotation = np.degrees(np.arctan2(matrix[1], matrix[0]))
                    char_rotations.append(char_rotation)

                # 提取颜色和透明度信息
                if hasattr(char, "graphicstate"):
                    if hasattr(char.graphicstate, "ncolor"):
                        color = char.graphicstate.ncolor
                        features["color"] = color
                        if len(color) >= 3:  # RGB颜色
                            avg_color = sum(color) / len(color)
                            if avg_color > 0.7:  # 浅色文本
                                features["light_color"] = True
                                confidence += 0.2
                                logger.debug(f"文本 '{text}' 为浅色文本")

                    # 提取透明度
                    if hasattr(char.graphicstate, "ca"):
                        opacity = float(char.graphicstate.ca)
                        opacity_values.append(opacity)
                        if opacity < 0.9:  # 半透明文本通常是水印
                            confidence += 0.25
                            logger.debug(f"文本 '{text}' 为半透明文本(透明度: {opacity:.2f})")

        # 分析旋转信息
        if char_rotations:
            avg_rotation = sum(char_rotations) / len(char_rotations)
            features["rotation"] = avg_rotation

            # 检查是否有一致的倾斜角度
            if 15 < abs(avg_rotation) < 75:
                confidence += 0.3
                logger.debug(f"文本 '{text}' 有明显倾斜角度: {avg_rotation:.1f}°")

        # 分析字体信息
        if font_sizes:
            avg_font_size = sum(font_sizes) / len(font_sizes)
            features["font_size"] = avg_font_size

            # 大号字体更可能是水印
            if avg_font_size > 14:
                confidence += 0.1
                logger.debug(f"文本 '{text}' 使用较大字体: {avg_font_size:.1f}pt")

        if font_names:
            features["font_name"] = list(font_names)

        # 分析透明度
        if opacity_values:
            features["opacity"] = sum(opacity_values) / len(opacity_values)

        # 特征4: 短文本更可能是水印
        if len(text) < 30:
            confidence += 0.1
        else:
            # 长文本通常不是水印
            confidence -= 0.2

        # 特征5: 重复单个字符的文本更可能是水印(如 "机密机密机密")
        if len(text) > 3 and len(set(text)) < len(text) / 2:
            confidence += 0.15
            features["repeated_pattern"] = True
            logger.debug(f"文本 '{text}' 包含重复模式")

        # 判断是否为水印
        is_watermark = confidence >= 0.5

        if is_watermark:
            logger.debug(f"识别为水印: '{text}', 置信度: {confidence:.2f}")

        return is_watermark, confidence, features

    def is_image_watermark(self, image_obj: Object, page: Object) -> tuple[bool, float, dict[str, Any]]:
        """判断图片元素是否为水印

        Args:
            image_obj: PDF图像对象
            page: PDF页面对象

        Returns:
            (是否为水印, 置信度, 特征字典)
        """
        confidence = 0.0
        features = {}

        # 特征1: 检查是否有透明度掩码
        if image_obj.get("/SMask") is not None:
            confidence += 0.4
            features["has_smask"] = True

        if image_obj.get("/Mask") is not None:
            confidence += 0.3
            features["has_mask"] = True

        # 特征2: 检查图像尺寸和位置
        if "/Width" in image_obj and "/Height" in image_obj:
            width = int(image_obj["/Width"])
            height = int(image_obj["/Height"])
            features["dimensions"] = (width, height)

            # 如果页面有MediaBox，检查图像是否覆盖整页
            if "/MediaBox" in page:
                media_box = page["/MediaBox"]
                page_width = float(media_box[2]) - float(media_box[0])
                page_height = float(media_box[3]) - float(media_box[1])

                # 计算图像与页面的比例
                width_ratio = width / page_width
                height_ratio = height / page_height
                features["coverage_ratio"] = (width_ratio, height_ratio)

                if width_ratio > 0.8 and height_ratio > 0.8:
                    confidence += 0.4
                    features["covers_page"] = True

        # 特征3: 检查图像颜色空间
        try:
            # 尝试提取图像
            pdfimage = PdfImage(image_obj)
            if hasattr(pdfimage, "colorspace"):
                color_space = str(pdfimage.colorspace)
                features["colorspace"] = color_space

                # 某些特定的颜色空间更可能用于水印
                if "DeviceGray" in color_space:
                    confidence += 0.2
                    features["grayscale"] = True
                elif "DeviceRGB" in color_space:
                    confidence += 0.1
                    features["rgb"] = True

        except Exception:
            pass

        # 判断是否为水印
        is_watermark = confidence >= 0.5

        return is_watermark, confidence, features

    def remove_watermark(self, input_pdf: str, output_pdf: str) -> dict[str, Any]:
        """水印去除函数

        Args:
            input_pdf: 输入PDF文件路径
            output_pdf: 输出PDF文件路径

        Returns:
            处理结果摘要的JSON对象
        """
        result = {"success": False, "input_file": input_pdf, "output_file": output_pdf, "watermarks_removed": []}

        try:
            # 1. 检测水印
            logger.info(f"开始检测水印: {input_pdf}")
            watermark_info = self.detect_watermark(input_pdf)

            if not watermark_info.get("has_watermark", False):
                logger.info(f"未在 {input_pdf} 中检测到水印")
                shutil.copy2(input_pdf, output_pdf)
                result["success"] = True
                result["message"] = "未检测到水印"
                return result

            # 创建临时文件用于中间处理
            temp_dir = tempfile.mkdtemp()
            self._temp_dirs.append(temp_dir)

            temp_pdf = os.path.join(temp_dir, "temp_processing.pdf")
            self._temp_files.append(temp_pdf)

            shutil.copy2(input_pdf, temp_pdf)
            current_pdf = temp_pdf

            # 2. 执行水印移除，按照类型依次处理
            if "text" in watermark_info.get("watermark_types", []):
                # 从检测结果中提取文本水印信息
                text_watermarks = watermark_info.get("text_watermarks_info", {}).get("text_watermarks", [])
                text_watermark_info = watermark_info.get("text_watermarks_info", {})

                if text_watermarks:
                    logger.info(f"检测到 {len(text_watermarks)} 种文本水印:")
                    for text in text_watermarks:
                        count = text_watermark_info.get("text_frequency", {}).get(text, 0)
                        logger.info(f"  - 水印文本 '{text}' 出现 {count} 次")

                # 提取结构化信息用于水印去除
                text_info = {
                    "text_watermarks": text_watermarks,
                    "watermarks_info": text_watermark_info.get("watermarks_info", []),
                    "watermark_by_page": text_watermark_info.get("watermark_by_page", {}),
                    "main_rotation": text_watermark_info.get("main_rotation", 0),
                    "rotation_groups": text_watermark_info.get("rotation_groups", {}),
                    "content_objects_info": watermark_info.get("watermark_details", {}).get("content_objects_info", []),
                    "xobject_info": watermark_info.get("watermark_details", {}).get("xobject_info", []),
                }

                logger.info(f"开始移除文本水印，共 {len(text_watermarks)} 种水印文本")
                temp_output = os.path.join(temp_dir, "text_removed.pdf")
                self._temp_files.append(temp_output)

                if self.remove_element(current_pdf, temp_output, "text", text_info):
                    result["watermarks_removed"].append("text")
                    current_pdf = temp_output

            if "image" in watermark_info.get("watermark_types", []):
                image_info = {
                    "image_watermarks": watermark_info.get("watermark_details", {}).get("image_watermarks", [])
                }

                logger.info(f"开始移除图片水印，共 {len(image_info['image_watermarks'])} 个图片水印")
                temp_output = os.path.join(temp_dir, "image_removed.pdf")
                self._temp_files.append(temp_output)

                if self.remove_element(current_pdf, temp_output, "image", image_info):
                    result["watermarks_removed"].append("image")
                    current_pdf = temp_output

            # 3. 复制最终结果到输出文件
            shutil.copy2(current_pdf, output_pdf)
            result["success"] = len(result["watermarks_removed"]) > 0

            if result["success"]:
                logger.info(f"成功移除水印类型: {result['watermarks_removed']}")
            else:
                logger.warning("未能移除任何水印")

            return result

        except Exception as e:
            logger.error(f"水印移除过程中发生错误: {e}")
            # 确保输出文件存在
            if not os.path.exists(output_pdf) and os.path.exists(input_pdf):
                try:
                    shutil.copy2(input_pdf, output_pdf)
                except Exception:
                    pass
            result["error"] = str(e)
            return result

    def remove_element(
        self,
        input_pdf: str,
        output_pdf: str,
        element_type: str,
        element_info: dict[str, Any],
    ) -> bool:
        """从PDF中移除指定元素

        Args:
            input_pdf: 输入PDF路径
            output_pdf: 输出PDF路径
            element_type: 元素类型 ('text' 或 'image')
            element_info: 元素信息字典

        Returns:
            成功返回True，失败返回False
        """
        logger.info(f"开始移除 {element_type} 元素")

        try:
            # 打开PDF并分析结构
            with pikepdf.open(input_pdf) as pdf:
                num_pages = len(pdf.pages)
                logger.info(f"PDF共有 {num_pages} 页")

                # 创建临时文件，以便在失败时恢复
                temp_dir = tempfile.mkdtemp()
                self._temp_dirs.append(temp_dir)

                temp_pdf = os.path.join(temp_dir, "temp_before_removal.pdf")
                self._temp_files.append(temp_pdf)

                pdf.save(temp_pdf)

                # 为不同类型元素选择适当的策略
                strategies = []
                if element_type == "text":
                    strategies = [self._strategy_replace_content_stream, self._strategy_remove_last_object]
                elif element_type == "image":
                    strategies = [self._strategy_remove_xobjects, self._strategy_remove_last_object]
                else:
                    logger.warning(f"未知元素类型: {element_type}")
                    return False

                # 依次尝试各种策略
                for strategy_idx, strategy in enumerate(strategies):
                    logger.info(f"尝试策略 {strategy_idx + 1}: {strategy.__name__}")

                    # 如果不是第一个策略，重新加载PDF
                    if strategy_idx > 0:
                        pdf.close()
                        pdf = pikepdf.open(temp_pdf)

                    # 应用策略
                    modified = strategy(pdf, element_info)

                    if modified:
                        pdf.save(output_pdf)
                        logger.info(f"策略 {strategy.__name__} 成功")
                        return True

                # 如果所有策略都失败
                logger.warning(f"未能移除任何{element_type}水印")
                # 确保输出文件存在
                shutil.copy2(input_pdf, output_pdf)
                return False

        except Exception as e:
            logger.error(f"移除{element_type}元素时出错: {e}")
            # 确保输出文件存在
            shutil.copy2(input_pdf, output_pdf)
            return False

    def _strategy_remove_last_object(self, pdf, element_info):
        """策略3: 删除最后一个内容对象"""
        modified = False
        for page_idx, page in enumerate(pdf.pages):
            if "/Contents" in page:
                contents = page["/Contents"]
                if isinstance(contents, list) and len(contents) > 1:
                    try:
                        new_contents = pikepdf.Array(contents[:-1])
                        page[Name.Contents] = new_contents
                        modified = True
                        logger.info(f"第{page_idx + 1}页: 删除最后一个内容对象")
                    except Exception as e:
                        logger.warning(f"删除最后一个内容对象失败: {e}")

        if modified:
            logger.info("策略3成功: 删除最后一个内容对象")
        return modified

    def _strategy_remove_xobjects(self, pdf, element_info):
        """策略2: 删除XObject"""
        modified = False
        removed_forms = 0
        removed_images = 0

        image_watermarks = element_info.get("image_watermarks", [])
        for image_element in image_watermarks:
            page = pdf.pages[image_element.get("page", 1) - 1]
            image_key = image_element.get("image_key")
            xobjects = page["/Resources"]["/XObject"]
            if image_key in xobjects.keys():
                del xobjects[image_key]
                modified = True
                if "/Form" in image_key:
                    removed_forms += 1
                else:
                    removed_images += 1

        if modified:
            logger.info(f"策略2成功: 删除了 {removed_forms} 个Form XObject和 {removed_images} 个Image XObject")
        return modified

    def _replace_watermark_in_stream(self, stream, watermark_info=None):
        """替换内容流中的水印文本

        Args:
            stream: PDF内容流字节数据
            watermark_info: 水印信息字典，包含要删除的水印特征

        Returns:
            (修改后的流, 是否被修改, 应用的替换规则列表)
        """
        original_stream = stream
        replacements_applied = []

        # 基本替换规则
        replacements = []

        # 如果有水印信息，添加针对性的替换规则
        if watermark_info:
            # 获取水印文本列表
            texts = watermark_info.get("text_watermarks", [])

            # 针对特定文本构建替换规则
            for text in texts:
                # 转义正则表达式特殊字符
                escaped_text = re.escape(text)
                # 添加针对特定文本的替换规则
                replacements.append((f"\\({escaped_text}\\)\\s*Tj".encode(), b"() Tj"))
                # 针对TJ操作符的规则
                replacements.append((f"\\[\\({escaped_text}\\)\\]\\s*TJ".encode(), b"[] TJ"))

            # 针对特定角度的文本添加规则
            if "main_rotation" in watermark_info:
                rotation = watermark_info["main_rotation"]
                # 如果文本有明显倾斜角度，添加针对旋转矩阵的替换
                if abs(rotation) > 5:
                    rot_sin = round(np.sin(np.radians(rotation)), 2)
                    rot_cos = round(np.cos(np.radians(rotation)), 2)

                    # 匹配旋转矩阵的模式
                    rot_pattern = f"{rot_cos}\\s+{rot_sin}\\s+{-rot_sin}\\s+{rot_cos}\\s+Tm".encode()
                    replacements.append((rot_pattern, b"1 0 0 1 Tm"))

                    logger.info(f"添加针对 {rotation}° 角度的替换规则")

        # 添加通用替换规则
        replacements.extend(
            [
                # 英文水印
                (rb"\(.*?(?:CONFIDENTIAL|DRAFT|INTERNAL).*?\)\s*Tj", b"() Tj"),
                # 常见PDF文本操作符
                (rb"BT\s+.*?ET", b"BT ET"),  # 简化文本块
                (rb"q\s+.*?Q", b"q Q"),  # 简化图形状态
                # 删除所有Tj操作符（通常用于渲染文本）
                (rb"\([^\)]*\)\s*Tj", b"() Tj"),
                # 删除所有TJ操作符（通常用于渲染文本）
                (rb"\[\([^\)]*\)\]\s*TJ", b"[] TJ"),
            ]
        )

        # 应用替换规则
        for i, (pattern, replacement) in enumerate(replacements):
            try:
                new_stream = re.sub(pattern, replacement, stream)
                if new_stream != stream:
                    if i < len(watermark_info.get("text_watermarks", [])) * 2:
                        replacements_applied.append(
                            f"删除水印文本: {watermark_info.get('text_watermarks', [])[i // 2]}"
                        )
                    else:
                        replacements_applied.append(f"规则{i + 1}")
                    stream = new_stream
            except Exception as e:
                logger.warning(f"应用替换规则 {i} 时出错: {e}")

        # 特殊处理：替换所有可能的文本渲染操作
        try:
            new_stream = re.sub(rb"\([^\)]{2,}\)\s*Tj", b"() Tj", stream)
            if new_stream != stream:
                replacements_applied.append("特殊规则: 删除所有文本渲染")
                stream = new_stream
        except Exception as e:
            logger.warning(f"应用特殊替换规则时出错: {e}")

        is_modified = stream != original_stream
        return stream, is_modified, replacements_applied

    def _strategy_replace_content_stream(self, pdf, element_info):
        """策略1: 替换内容流中的水印文本"""
        modified = False
        pages_modified = 0
        objects_modified = 0

        # 遍历所有页面
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1
            if "/Contents" in page:
                contents = page["/Contents"]
                page_modified = False

                # 处理单个内容对象
                if not isinstance(contents, list):
                    try:
                        stream = contents.read_bytes()
                        new_stream, is_modified, replacements_applied = self._replace_watermark_in_stream(
                            stream, element_info
                        )

                        if is_modified:
                            contents.write(new_stream)
                            modified = True
                            page_modified = True
                            objects_modified += 1
                            logger.info(
                                f"第{page_num}页: 替换了内容流中的水印文本，应用规则: {', '.join(replacements_applied)}"
                            )
                    except Exception as e:
                        logger.warning(f"替换内容流失败: {e}")

                # 处理多个内容对象
                elif isinstance(contents, list):
                    for i, content_obj in enumerate(contents):
                        try:
                            stream = content_obj.read_bytes()
                            new_stream, is_modified, replacements_applied = self._replace_watermark_in_stream(
                                stream, element_info
                            )

                            if is_modified:
                                content_obj.write(new_stream)
                                modified = True
                                page_modified = True
                                objects_modified += 1
                                logger.info(
                                    f"第{page_num}页内容对象{i}: 替换了内容流中的水印文本，应用规则: {', '.join(replacements_applied)}"
                                )
                        except Exception as e:
                            logger.warning(f"替换内容对象{i}的内容流失败: {e}")

                if page_modified:
                    pages_modified += 1

        if modified:
            logger.info(f"策略1成功: 替换了 {pages_modified} 页中的 {objects_modified} 个内容对象")
        return modified
