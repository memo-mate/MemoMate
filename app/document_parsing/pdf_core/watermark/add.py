import os
import tempfile

from PIL import Image
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.colors import Color, black, blue, green, grey, red, white
from reportlab.pdfgen import canvas

from app.core import logger


class WatermarkAdder:
    """PDF水印添加器"""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    def add_text_watermark(
        self,
        input_pdf: str,
        output_pdf: str,
        text: str,
        font_size: int = 50,
        color: str | tuple[float, float, float] = (0.5, 0.5, 0.5),  # 灰色半透明
        opacity: float = 0.3,
        angle: int = 45,
        position: str = "center",
        pattern: str = "normal",  # 水印模式: normal, sparse, dense, tiled
        spacing: int = 100,  # 平铺模式下的间距
        pages: list[int] | None = None,
    ) -> bool:
        """添加文本水印

        Args:
            input_pdf: 输入PDF路径
            output_pdf: 输出PDF路径
            text: 水印文本
            font_size: 字体大小
            color: 水印颜色，可以是(r,g,b)元组或颜色名称字符串
            opacity: 不透明度，0-1之间
            angle: 旋转角度
            position: 位置，可选 "center", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right"
            pattern: 水印模式，可选 "sparse"(稀疏), "normal"(正常), "dense"(密集), "tiled"(平铺)
            spacing: 平铺模式下的水印间距
            pages: 指定页码列表，None表示所有页面

        Returns:
            成功返回True，失败返回False
        """
        try:
            # 转换颜色格式
            if isinstance(color, tuple):
                r, g, b = color
                color_obj = Color(r, g, b, alpha=opacity)
            else:
                # 预定义颜色
                color_map = {
                    "black": black,
                    "white": white,
                    "red": red,
                    "blue": blue,
                    "green": green,
                    "grey": grey,
                    "gray": grey,
                }
                color_obj = color_map.get(color.lower(), grey)
                # 应用透明度
                color_obj = Color(color_obj.red, color_obj.green, color_obj.blue, alpha=opacity)

            # 创建一个临时水印PDF
            watermark_pdf = os.path.join(self.temp_dir, "watermark.pdf")

            # 读取原始PDF获取页面尺寸
            reader = PdfReader(input_pdf)
            num_pages = len(reader.pages)

            if pages is None:
                pages = list(range(num_pages))
            else:
                pages = [p - 1 for p in pages if 1 <= p <= num_pages]  # 转换为0-based索引

            if not pages:
                logger.error("没有有效的页面需要添加水印")
                return False

            # 获取第一页的尺寸作为参考
            page = reader.pages[0]
            page_width = float(page.mediabox.width)
            page_height = float(page.mediabox.height)

            # 创建水印
            c = canvas.Canvas(watermark_pdf, pagesize=(page_width, page_height))
            c.setFillColor(color_obj)

            # 检查文本是否包含中文字符
            def contains_chinese(text):
                for char in text:
                    if "\u4e00" <= char <= "\u9fff":
                        return True
                return False

            # 设置字体，如果包含中文则使用中文字体
            if contains_chinese(text):
                try:
                    # 尝试使用系统中文字体
                    from reportlab.pdfbase import pdfmetrics
                    from reportlab.pdfbase.ttfonts import TTFont

                    # 尝试常见的中文字体路径
                    font_paths = [
                        os.path.join(os.environ.get("WINDIR", ""), "Fonts", "simhei.ttf"),  # Windows
                        os.path.join(os.environ.get("WINDIR", ""), "Fonts", "simsun.ttc"),  # Windows
                        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
                        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",  # Linux
                        "/System/Library/Fonts/PingFang.ttc",  # macOS
                        "/System/Library/Fonts/STHeiti Light.ttc",  # macOS
                    ]

                    font_found = False
                    for font_path in font_paths:
                        if os.path.exists(font_path):
                            try:
                                pdfmetrics.registerFont(TTFont("ChineseFont", font_path))
                                c.setFont("ChineseFont", font_size)
                                font_found = True
                                break
                            except Exception as e:
                                logger.error(f"注册字体失败: {e}")
                                continue

                    if not font_found:
                        # 如果没有找到中文字体，使用默认字体
                        logger.error("未找到中文字体，使用默认字体")
                        c.setFont("Helvetica", font_size)
                except Exception as e:
                    logger.error(f"加载中文字体时出错: {e}")
                    c.setFont("Helvetica", font_size)
            else:
                c.setFont("Helvetica", font_size)

            # 根据模式绘制水印
            if pattern == "tiled":
                # 平铺模式
                rows = int(page_height / spacing) + 2
                cols = int(page_width / spacing) + 2

                # 绘制平铺水印
                for i in range(rows):
                    for j in range(cols):
                        x = j * spacing
                        y = i * spacing
                        c.saveState()
                        c.translate(x, y)
                        c.rotate(angle)
                        c.drawCentredString(0, 0, text)
                        c.restoreState()
            else:
                # 非平铺模式（sparse, normal, dense）
                # 根据密度确定水印数量
                density_map = {"sparse": (1, 1), "normal": (3, 3), "dense": (5, 5)}
                rows, cols = density_map.get(pattern.lower(), (3, 3))

                # 计算水印位置
                horizontal_step = page_width / (cols + 1)
                vertical_step = page_height / (rows + 1)

                # 根据position调整起始位置
                position_map = {
                    "center": (page_width / 2, page_height / 2),
                    "top": (page_width / 2, page_height * 0.8),
                    "bottom": (page_width / 2, page_height * 0.2),
                    "left": (page_width * 0.2, page_height / 2),
                    "right": (page_width * 0.8, page_height / 2),
                    "top-left": (page_width * 0.2, page_height * 0.8),
                    "top-right": (page_width * 0.8, page_height * 0.8),
                    "bottom-left": (page_width * 0.2, page_height * 0.2),
                    "bottom-right": (page_width * 0.8, page_height * 0.2),
                }

                if pattern == "sparse":
                    # 稀疏模式只在指定位置添加一个水印
                    x, y = position_map.get(position.lower(), (page_width / 2, page_height / 2))
                    c.saveState()
                    c.translate(x, y)
                    c.rotate(angle)
                    c.drawCentredString(0, 0, text)
                    c.restoreState()
                else:
                    # 正常或密集模式在多个位置添加水印
                    for i in range(1, rows + 1):
                        for j in range(1, cols + 1):
                            x = j * horizontal_step
                            y = i * vertical_step
                            c.saveState()
                            c.translate(x, y)
                            c.rotate(angle)
                            c.drawCentredString(0, 0, text)
                            c.restoreState()

            c.save()

            # 将水印应用到原始PDF
            watermark_reader = PdfReader(watermark_pdf)
            watermark_page = watermark_reader.pages[0]

            writer = PdfWriter()

            # 处理每一页
            for i in range(num_pages):
                if i in pages:
                    page = reader.pages[i]
                    page.merge_page(watermark_page)
                    writer.add_page(page)
                else:
                    writer.add_page(reader.pages[i])

            # 保存结果
            with open(output_pdf, "wb") as output_file:
                writer.write(output_file)

            return True

        except Exception as e:
            logger.error(f"添加文本水印时出错: {e}")
            return False

    def add_image_watermark(
        self,
        input_pdf: str,
        output_pdf: str,
        image_path: str,
        scale: float = 0.3,
        opacity: float = 0.3,
        position: str = "center",
        pages: list[int] | None = None,
    ) -> bool:
        """添加图片水印

        Args:
            input_pdf: 输入PDF路径
            output_pdf: 输出PDF路径
            image_path: 水印图片路径
            scale: 图片缩放比例，相对于页面宽度
            opacity: 不透明度，0-1之间
            position: 位置，可选 "center", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right"
            pages: 指定页码列表，None表示所有页面

        Returns:
            成功返回True，失败返回False
        """
        try:
            # 打开水印图片并调整透明度
            img = Image.open(image_path)
            img = img.convert("RGBA")

            # 创建一个临时水印PDF
            watermark_pdf = os.path.join(self.temp_dir, "image_watermark.pdf")

            # 读取原始PDF获取页面尺寸
            reader = PdfReader(input_pdf)
            num_pages = len(reader.pages)

            if pages is None:
                pages = list(range(num_pages))
            else:
                pages = [p - 1 for p in pages if 1 <= p <= num_pages]  # 转换为0-based索引

            if not pages:
                logger.error("没有有效的页面需要添加水印")
                return False

            # 获取第一页的尺寸作为参考
            page = reader.pages[0]
            page_width = float(page.mediabox.width)
            page_height = float(page.mediabox.height)

            # 计算水印图片的尺寸
            img_width = int(page_width * scale)
            img_height = int(img.height * (img_width / img.width))
            img_resized = img.resize((img_width, img_height), Image.LANCZOS)

            # 应用透明度
            data = img_resized.getdata()
            new_data = []
            for item in data:
                # 保持RGB值不变，只修改alpha通道
                if len(item) == 4:  # RGBA
                    new_data.append((item[0], item[1], item[2], int(item[3] * opacity)))
                else:  # RGB
                    new_data.append((item[0], item[1], item[2], int(255 * opacity)))

            img_resized.putdata(new_data)

            # 计算水印位置
            position_map = {
                "center": ((page_width - img_width) / 2, (page_height - img_height) / 2),
                "top": ((page_width - img_width) / 2, page_height - img_height - 50),
                "bottom": ((page_width - img_width) / 2, 50),
                "left": (50, (page_height - img_height) / 2),
                "right": (page_width - img_width - 50, (page_height - img_height) / 2),
                "top-left": (50, page_height - img_height - 50),
                "top-right": (page_width - img_width - 50, page_height - img_height - 50),
                "bottom-left": (50, 50),
                "bottom-right": (page_width - img_width - 50, 50),
            }

            x, y = position_map.get(position.lower(), ((page_width - img_width) / 2, (page_height - img_height) / 2))

            # 创建水印PDF
            c = canvas.Canvas(watermark_pdf, pagesize=(page_width, page_height))

            # 保存图片到临时文件
            img_temp_path = os.path.join(self.temp_dir, "watermark_img_temp.png")
            print(img_temp_path)
            img_resized.save(img_temp_path, "PNG")

            # 添加图片到canvas
            c.drawImage(img_temp_path, x, y, width=img_width, height=img_height, mask="auto")
            c.save()

            # 将水印应用到原始PDF
            watermark_reader = PdfReader(watermark_pdf)
            watermark_page = watermark_reader.pages[0]

            writer = PdfWriter()

            # 处理每一页
            for i in range(num_pages):
                if i in pages:
                    page = reader.pages[i]
                    page.merge_page(watermark_page)
                    writer.add_page(page)
                else:
                    writer.add_page(reader.pages[i])

            # 保存结果
            with open(output_pdf, "wb") as output_file:
                writer.write(output_file)

            return True

        except Exception as e:
            logger.error(f"添加图片水印时出错: {e}")
            return False
