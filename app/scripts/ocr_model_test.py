import requests
import os
from io import BytesIO
from dotenv import load_dotenv
import base64
import urllib3


# 加载环境变量
load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# 配置参数
API_KEY = os.getenv("OPENAI_API_KEY")

EMPTY_PROXIES = {
    "http": "",
    "https": "",
}


def process_image(file_path, model="Qwen/Qwen2-VL-72B-Instruct"):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    # 转成IO流
    buffer = BytesIO()
    buffer.write(open(file_path, "rb").read())  # 写入二进制数据
    byte_data = buffer.getvalue()

    # 将图片字节编码为base64
    encoded_image = base64.b64encode(byte_data).decode("utf-8")

    print(f"使用模型：{model}")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "将图片内容尽可能描述清晰，并转换为Markdown格式，保留表格和公式结构，用中文输出，不要输出任何解释",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ],
            }
        ],
        "temperature": 0.1,
    }

    response = requests.post(
        "https://api.siliconflow.cn/v1/chat/completions",
        headers=headers,
        json=payload,
        verify=False,
        proxies=EMPTY_PROXIES,
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API Error: {response.text}")


if __name__ == "__main__":
    try:
        file_path = r"C:\Users\STARTER\Pictures\计算机性能指标.png"

        markdown_content = process_image(file_path, model="deepseek-ai/deepseek-vl2")
        print(markdown_content)

    except Exception as e:
        print(f"处理失败: {str(e)}")
