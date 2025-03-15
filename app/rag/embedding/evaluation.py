import time
from typing import Any

from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel


class RetrievalEvaluator(BaseModel):
    """检索评估器"""

    retriever: BaseRetriever

    def evaluate_precision(
        self,
        queries: list[str],
        relevant_docs: list[list[str]],
        k: int = 5,
    ) -> dict[str, Any]:
        """评估精确率"""
        results = []
        total_precision = 0.0

        for i, query in enumerate(queries):
            # 获取检索结果
            retrieved_docs = self.retriever.invoke(query)[:k]
            retrieved_contents = [doc.page_content for doc in retrieved_docs]

            # 计算精确率
            relevant = relevant_docs[i] if i < len(relevant_docs) else []
            relevant_retrieved = [content for content in retrieved_contents if content in relevant]
            precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0.0

            # 添加结果
            results.append(
                {
                    "query": query,
                    "precision": precision,
                    "retrieved_count": len(retrieved_docs),
                    "relevant_retrieved_count": len(relevant_retrieved),
                }
            )

            total_precision += precision

        # 计算平均精确率
        avg_precision = total_precision / len(queries) if queries else 0.0

        return {
            "avg_precision": avg_precision,
            "results": results,
        }

    def evaluate_recall(
        self,
        queries: list[str],
        relevant_docs: list[list[str]],
        k: int = 5,
    ) -> dict[str, Any]:
        """评估召回率"""
        results = []
        total_recall = 0.0

        for i, query in enumerate(queries):
            # 获取检索结果
            retrieved_docs = self.retriever.invoke(query)[:k]
            retrieved_contents = [doc.page_content for doc in retrieved_docs]

            # 计算召回率
            relevant = relevant_docs[i] if i < len(relevant_docs) else []
            relevant_retrieved = [content for content in retrieved_contents if content in relevant]
            recall = len(relevant_retrieved) / len(relevant) if relevant else 0.0

            # 添加结果
            results.append(
                {
                    "query": query,
                    "recall": recall,
                    "retrieved_count": len(retrieved_docs),
                    "relevant_count": len(relevant),
                    "relevant_retrieved_count": len(relevant_retrieved),
                }
            )

            total_recall += recall

        # 计算平均召回率
        avg_recall = total_recall / len(queries) if queries else 0.0

        return {
            "avg_recall": avg_recall,
            "results": results,
        }

    def evaluate_latency(
        self,
        queries: list[str],
        k: int = 5,
        runs: int = 3,
    ) -> dict[str, Any]:
        """评估延迟"""
        results = []
        total_latency = 0.0

        for query in queries:
            query_latencies = []

            for _ in range(runs):
                # 测量检索时间
                start_time = time.time()
                self.retriever.invoke(query)[:k]
                end_time = time.time()

                latency = end_time - start_time
                query_latencies.append(latency)

            # 计算平均延迟
            avg_query_latency = sum(query_latencies) / len(query_latencies)

            # 添加结果
            results.append(
                {
                    "query": query,
                    "avg_latency": avg_query_latency,
                    "min_latency": min(query_latencies),
                    "max_latency": max(query_latencies),
                }
            )

            total_latency += avg_query_latency

        # 计算总平均延迟
        avg_latency = total_latency / len(queries) if queries else 0.0

        return {
            "avg_latency": avg_latency,
            "results": results,
        }

    def evaluate_all(
        self,
        queries: list[str],
        relevant_docs: list[list[str]],
        k: int = 5,
        runs: int = 3,
    ) -> dict[str, Any]:
        """评估所有指标"""
        precision_results = self.evaluate_precision(queries, relevant_docs, k)
        recall_results = self.evaluate_recall(queries, relevant_docs, k)
        latency_results = self.evaluate_latency(queries, k, runs)

        # 计算F1分数
        f1_results = []
        total_f1 = 0.0

        for i in range(len(queries)):
            precision = precision_results["results"][i]["precision"]
            recall = recall_results["results"][i]["recall"]

            # 计算F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            f1_results.append(
                {
                    "query": queries[i],
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                }
            )

            total_f1 += f1

        # 计算平均F1分数
        avg_f1 = total_f1 / len(queries) if queries else 0.0

        return {
            "precision": precision_results["avg_precision"],
            "recall": recall_results["avg_recall"],
            "f1": avg_f1,
            "latency": latency_results["avg_latency"],
            "detail": {
                "precision": precision_results["results"],
                "recall": recall_results["results"],
                "f1": f1_results,
                "latency": latency_results["results"],
            },
        }


class RAGEvaluator(BaseModel):
    """RAG评估器"""

    retriever: BaseRetriever
    llm: BaseLLM

    def evaluate_answer_relevance(
        self,
        queries: list[str],
        ground_truth: list[str],
    ) -> dict[str, Any]:
        """评估回答相关性"""
        from langchain_core.prompts import ChatPromptTemplate

        results = []
        total_score = 0.0

        # 创建评估提示
        eval_prompt = ChatPromptTemplate.from_template(
            """你是一个评估AI回答质量的专家。请评估以下AI回答与参考答案的相关性。
            
            问题: {question}
            AI回答: {answer}
            参考答案: {reference}
            
            请给出1-5的评分，其中:
            1: 完全不相关
            2: 略微相关
            3: 部分相关
            4: 大部分相关
            5: 完全相关
            
            只需返回评分数字，不要有其他文本。
            """
        )

        for i, query in enumerate(queries):
            # 获取检索结果
            docs = self.retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            # 生成回答
            answer_prompt = ChatPromptTemplate.from_template(
                """回答以下问题，基于提供的上下文信息。如果无法从上下文中找到答案，请说"我不知道"。
                
                上下文: {context}
                问题: {question}
                
                回答:"""
            )

            answer_chain = answer_prompt | self.llm
            answer = answer_chain.invoke({"context": context, "question": query}).content

            # 评估回答
            reference = ground_truth[i] if i < len(ground_truth) else ""
            eval_chain = eval_prompt | self.llm

            try:
                score_text = eval_chain.invoke(
                    {"question": query, "answer": answer, "reference": reference}
                ).content
                score = float(score_text.strip())
            except ValueError:
                # 如果无法解析分数，默认为0
                score = 0.0

            # 添加结果
            results.append(
                {
                    "query": query,
                    "answer": answer,
                    "reference": reference,
                    "score": score,
                }
            )

            total_score += score

        # 计算平均分数
        avg_score = total_score / len(queries) if queries else 0.0

        return {
            "avg_score": avg_score,
            "results": results,
        }

    def evaluate_hallucination(
        self,
        queries: list[str],
    ) -> dict[str, Any]:
        """评估幻觉"""
        from langchain_core.prompts import ChatPromptTemplate

        results = []
        total_score = 0.0

        # 创建评估提示
        eval_prompt = ChatPromptTemplate.from_template(
            """你是一个评估AI回答质量的专家。请评估以下AI回答是否存在幻觉（即生成了不在上下文中的信息）。
            
            问题: {question}
            上下文: {context}
            AI回答: {answer}
            
            请给出1-5的评分，其中:
            1: 严重幻觉，回答完全不基于上下文
            2: 明显幻觉，回答大部分不基于上下文
            3: 部分幻觉，回答部分不基于上下文
            4: 轻微幻觉，回答大部分基于上下文，但有少量不准确信息
            5: 无幻觉，回答完全基于上下文
            
            只需返回评分数字，不要有其他文本。
            """
        )

        for query in queries:
            # 获取检索结果
            docs = self.retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            # 生成回答
            answer_prompt = ChatPromptTemplate.from_template(
                """回答以下问题，基于提供的上下文信息。如果无法从上下文中找到答案，请说"我不知道"。
                
                上下文: {context}
                问题: {question}
                
                回答:"""
            )

            answer_chain = answer_prompt | self.llm
            answer = answer_chain.invoke({"context": context, "question": query}).content

            # 评估幻觉
            eval_chain = eval_prompt | self.llm

            try:
                score_text = eval_chain.invoke(
                    {"question": query, "context": context, "answer": answer}
                ).content
                score = float(score_text.strip())
            except ValueError:
                # 如果无法解析分数，默认为0
                score = 0.0

            # 添加结果
            results.append(
                {
                    "query": query,
                    "context": context,
                    "answer": answer,
                    "score": score,
                }
            )

            total_score += score

        # 计算平均分数
        avg_score = total_score / len(queries) if queries else 0.0

        return {
            "avg_score": avg_score,
            "results": results,
        }
