import os
import requests  # type: ignore
import urllib3
from langchain_core.prompts import ChatPromptTemplate  # noqa: F401
from app.core.vector_search import VectorSearch
from dotenv import load_dotenv
from loguru import logger

# 禁用不安全请求的警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 加载环境变量
load_dotenv()

# 禁用代理
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

EMPTY_PROXIES = {
    "http": "",
    "https": "",
}


class KnowledgeBaseQA:
    def __init__(
        self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", temperature=0.1
    ):
        # 初始化向量搜索
        self.vector_search = VectorSearch()

        # 设置OpenAI API参数
        self.api_base = os.getenv("OPENAI_API_BASE")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature

        # 定义提示模板
        self.prompt_template = """
        你是一个专业的助手，请基于以下检索到的信息回答用户的问题。
        如果检索到的信息无法回答用户的问题，请直接说明你不知道，不要编造答案。
        
        检索到的信息:
        {context}
        
        用户问题: {question}
        
        请给出详细、准确的回答，并引用相关的信息来源。
        """

    def _get_context(self, query):
        """从向量数据库获取相关上下文"""
        results = self.vector_search.similarity_search(query, k=20)

        # 打印检索结果，用于评估质量
        # print("\n===== 向量检索结果 =====")
        # for i, (doc, score) in enumerate(results):
        #     source = doc.metadata.get("source", "未知来源")
        #     page = doc.metadata.get("page", "")
        #     page_info = f"(第{page}页)" if page else ""

        #     print(f"[文档{i+1}] 相似度: {score:.4f}")
        #     print(f"来源: {source}{page_info}")
        #     print(
        #         f"内容: {doc.page_content[:150]}..."
        #         if len(doc.page_content) > 150
        #         else f"内容: {doc.page_content}"
        #     )
        #     print("-" * 50)
        # print("========================\n")

        # 格式化上下文
        context_texts = []
        for i, (doc, score) in enumerate(results):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            page_info = f"(第{page}页)" if page else ""

            context_text = f"[文档{i+1}] {source}{page_info}\n{doc.page_content}\n"
            context_texts.append(context_text)

        return "\n".join(context_texts)

    def _call_openai_api(self, messages):
        """调用OpenAI API"""
        # 禁用代理
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            verify=False,  # 禁用SSL验证，仅用于演示
            proxies=EMPTY_PROXIES,  # 显式设置空代理
        )

        if response.status_code != 200:
            raise Exception(f"API调用失败: {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    @logger.catch
    def answer_question(self, question):
        """回答用户问题"""
        # 获取相关上下文
        context = self._get_context(question)

        # 构建消息
        prompt = self.prompt_template.format(context=context, question=question)
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的助手，基于检索到的信息回答问题。",
            },
            {"role": "user", "content": prompt},
        ]

        # 调用API
        return self._call_openai_api(messages)


# 使用示例
if __name__ == "__main__":
    question = "根据2024年10月周报，总结一下10月的工作内容"
    print(f"提问：{question}\n")
    try:
        qa_system = KnowledgeBaseQA()
        answer = qa_system.answer_question(question)
        print("\n回答:")
        print(answer)
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
