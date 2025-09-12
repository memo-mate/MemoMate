from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.utils.web_search_summary import (
    DocumentFormat,
    HumanReviewRequiredException,
    ProcessingConfig,
    SummaryStrategy,
    WebSearchProcessor,
    summarize_text_list,
    summarize_web_content,
)


class TestWebSearchProcessor:
    """测试Web搜索处理器核心功能"""

    @pytest.fixture
    def processor(self):
        """创建处理器实例"""
        with patch("app.utils.web_search.create_llm") as mock_create_llm:
            mock_llm = MagicMock()
            mock_llm.get_num_tokens.return_value = 100
            mock_create_llm.return_value = mock_llm

            config = ProcessingConfig(enable_cache=False)
            processor = WebSearchProcessor(config)
            processor.llm = mock_llm
            return processor

    def test_get_strategy(self, processor):
        """测试获取摘要策略"""
        from app.utils.web_search_summary import ConciseStrategy

        strategy = processor.get_strategy(SummaryStrategy.CONCISE.value)
        assert isinstance(strategy, ConciseStrategy)

    def test_length_function(self, processor):
        """测试长度计算函数"""
        docs = [Document(page_content="test content 1"), Document(page_content="test content 2")]
        length = processor.length_function(docs)
        assert length == 200  # 2 * 100 (mocked return value)

    def test_length_function_fallback(self, processor):
        """测试长度计算函数的fallback逻辑"""
        processor.llm.get_num_tokens.side_effect = Exception("Token calculation failed")
        docs = [
            Document(page_content="test content one two three"),
            Document(page_content="test content four five six"),
        ]
        length = processor.length_function(docs)
        expected = (5 * 1.3) + (5 * 1.3)  # 每个文档5个单词，乘以1.3的估算系数
        assert length == expected

    async def test_generate_summary_failure(self, processor):
        """测试生成摘要失败处理"""
        with patch.object(processor, "get_strategy") as mock_get_strategy:
            mock_get_strategy.side_effect = Exception("Strategy error")

            state = {"content": "test content", "strategy": SummaryStrategy.CONCISE.value, "config": ProcessingConfig()}
            result = await processor.generate_summary(state)
            assert "摘要生成失败" in result["summaries"][0]

    def test_map_summaries(self, processor):
        """测试映射摘要"""
        state = {
            "contents": ["content1", "content2"],
            "config": ProcessingConfig(),
            "strategy": SummaryStrategy.CONCISE.value,
        }
        result = processor.map_summaries(state)
        assert len(result) == 2
        assert all(send.node == "generate_summary" for send in result)

    def test_collect_summaries(self, processor):
        """测试收集摘要"""
        state = {"summaries": ["summary1", "summary2"], "processing_steps": ["step1"]}
        result = processor.collect_summaries(state)
        assert len(result["collapsed_summaries"]) == 2
        assert "收集了 2 个摘要" in result["processing_steps"]

    def test_should_collapse_logic(self, processor):
        """测试折叠决策逻辑"""
        # 测试需要继续折叠的情况
        processor.length_function = MagicMock(return_value=5000)
        state = {"collapsed_summaries": [Document(page_content="test")], "config": ProcessingConfig(token_max=3000)}
        result = processor.should_collapse(state)
        assert result == "collapse_summaries"

        # 测试生成最终摘要的情况
        processor.length_function = MagicMock(return_value=1000)
        state = {"collapsed_summaries": [Document(page_content="test")], "config": ProcessingConfig(token_max=3000)}
        result = processor.should_collapse(state)
        assert result == "generate_final_summary"


class TestHumanReviewFeature:
    """测试人工审核功能"""

    @pytest.fixture
    def processor_with_human_review(self):
        """创建启用人工审核的处理器实例"""
        with patch("app.utils.web_search.create_llm") as mock_create_llm:
            mock_llm = MagicMock()
            mock_llm.get_num_tokens.return_value = 100
            mock_create_llm.return_value = mock_llm

            config = ProcessingConfig(enable_cache=False, enable_human_review=True, token_max=2000)
            processor = WebSearchProcessor(config)
            processor.llm = mock_llm
            return processor

    def test_should_collapse_triggers_human_review(self, processor_with_human_review):
        """测试should_collapse方法在适当条件下触发人工审核"""
        processor_with_human_review.length_function = MagicMock(return_value=1000)
        state = {
            "collapsed_summaries": [Document(page_content="test content")],
            "config": processor_with_human_review.config,
            "metadata": {},
        }
        result = processor_with_human_review.should_collapse(state)
        assert result == "human_review"

    async def test_human_review_no_handler_raises_exception(self, processor_with_human_review):
        """测试没有处理器时抛出异常"""
        state = {
            "collapsed_summaries": [Document(page_content="test content for review")],
            "metadata": {"existing": "data"},
        }
        with pytest.raises(HumanReviewRequiredException) as exc_info:
            await processor_with_human_review.human_review(state)

        assert "需要人工审核" in str(exc_info.value)
        assert exc_info.value.summary_preview
        assert exc_info.value.state == state

    async def test_human_review_with_custom_handler(self, processor_with_human_review):
        """测试自定义审核处理器"""

        async def custom_handler(state):
            return {"approved": True, "feedback": "自定义处理器批准", "reviewer": "custom_handler"}

        processor_with_human_review.set_human_review_handler(custom_handler)
        state = {
            "collapsed_summaries": [Document(page_content="test content for review")],
            "metadata": {"existing": "data"},
        }
        result = await processor_with_human_review.human_review(state)

        assert result["metadata"]["human_review_status"] == "completed"
        assert result["metadata"]["review_approved"] is True
        assert result["metadata"]["review_feedback"] == "自定义处理器批准"
        assert result["metadata"]["reviewer"] == "custom_handler"

    async def test_human_review_interactive_mode(self, processor_with_human_review):
        """测试交互式审核模式"""
        config = ProcessingConfig(enable_human_review=True, interactive_review=True)
        processor_with_human_review.config = config

        state = {
            "collapsed_summaries": [Document(page_content="test content for review")],
            "metadata": {"existing": "data"},
        }

        with patch("builtins.input", side_effect=["y", "测试批准"]):
            result = await processor_with_human_review.human_review(state)

        assert result["metadata"]["human_review_status"] == "completed"
        assert result["metadata"]["review_approved"] is True
        assert result["metadata"]["review_feedback"] == "测试批准"


class TestSummaryStrategies:
    """测试摘要策略"""

    def test_strategy_templates(self):
        """测试各种策略的模板"""
        from app.utils.web_search_summary import (
            BulletPointsStrategy,
            ConciseStrategy,
            DetailedStrategy,
            PromptTemplateStrategy,
            TechnicalStrategy,
        )

        strategies: list[tuple[PromptTemplateStrategy, str]] = [
            (ConciseStrategy(), "简洁"),
            (DetailedStrategy(), "详细"),
            (BulletPointsStrategy(), "要点"),
            (TechnicalStrategy(), "技术"),
        ]

        for strategy, keyword in strategies:
            assert keyword in strategy.get_map_template()
            assert keyword in strategy.get_reduce_template()


class TestDocumentLoaderFactory:
    """测试文档加载器工厂"""

    def test_web_loader(self):
        """测试Web加载器"""
        from app.utils.web_search_summary import DocumentLoaderFactory

        loader = DocumentLoaderFactory.create_loader("https://example.com", DocumentFormat.WEB)
        assert loader is not None

    def test_unsupported_format(self):
        """测试不支持的格式"""
        from app.utils.web_search_summary import DocumentLoaderFactory

        with pytest.raises(ValueError, match="不支持的文档格式"):
            DocumentLoaderFactory.create_loader("test", "unsupported")


class TestConvenienceFunctions:
    """测试便捷函数"""

    async def test_summarize_text_list(self):
        """测试文本列表摘要"""
        with patch("app.utils.web_search.WebSearchProcessor") as MockProcessor:
            mock_processor = AsyncMock()
            mock_processor.process_documents.return_value = {"final_summary": "测试摘要结果"}
            MockProcessor.return_value = mock_processor

            result = await summarize_text_list(["text1", "text2"])
            assert result == "测试摘要结果"
            MockProcessor.assert_called_once()

    async def test_summarize_web_content(self):
        """测试Web内容摘要"""
        with patch("app.utils.web_search.WebSearchProcessor") as MockProcessor:
            mock_processor = AsyncMock()
            mock_processor.process_documents.return_value = {"final_summary": "Web摘要结果"}
            MockProcessor.return_value = mock_processor

            result = await summarize_web_content("https://example.com")
            assert result == "Web摘要结果"
            MockProcessor.assert_called_once()


class TestHumanReviewException:
    """测试人工审核异常"""

    def test_human_review_required_exception_creation(self):
        """测试HumanReviewRequiredException异常创建"""
        message = "需要人工审核"
        summary_preview = "这是摘要预览..."
        state = {"test": "data"}

        exc = HumanReviewRequiredException(message, summary_preview, state)

        assert str(exc) == message
        assert exc.summary_preview == summary_preview
        assert exc.state == state

    def test_human_review_required_exception_without_optional_params(self):
        """测试不带可选参数的异常创建"""
        message = "需要人工审核"
        exc = HumanReviewRequiredException(message)

        assert str(exc) == message
        assert exc.summary_preview == ""
        assert exc.state is None


class TestWorkflowIntegration:
    """测试工作流程集成"""

    @pytest.fixture
    def processor_workflow(self):
        """创建工作流程测试处理器"""
        with patch("app.utils.web_search.create_llm") as mock_create_llm:
            mock_llm = MagicMock()
            mock_llm.get_num_tokens.return_value = 100
            mock_create_llm.return_value = mock_llm

            config = ProcessingConfig(enable_cache=False, enable_human_review=True, token_max=1500)
            processor = WebSearchProcessor(config)
            processor.llm = mock_llm
            return processor

    async def test_complete_workflow_with_human_review(self, processor_workflow):
        """测试包含人工审核的完整工作流程"""

        async def workflow_handler(state):
            return {"approved": True, "feedback": "工作流程测试通过"}

        processor_workflow.set_human_review_handler(workflow_handler)

        test_documents = [
            "第一部分：项目概述",
            "第二部分：技术方案",
            "第三部分：风险评估",
        ]

        # 1. 初始状态
        initial_state = {
            "contents": test_documents,
            "config": processor_workflow.config,
            "strategy": SummaryStrategy.CONCISE.value,
            "processing_steps": ["文档加载"],
        }

        # 2. 映射摘要
        send_results = processor_workflow.map_summaries(initial_state)
        assert len(send_results) == 3

        # 3. 收集摘要
        collected_state = {
            "summaries": [f"摘要: {doc[:15]}..." for doc in test_documents],
            "processing_steps": ["文档加载", "映射摘要"],
            "config": processor_workflow.config,
        }

        after_collect = processor_workflow.collect_summaries(collected_state)
        assert len(after_collect["collapsed_summaries"]) == 3

        # 4. 检查决策逻辑
        processor_workflow.length_function = MagicMock(return_value=1200)
        after_collect["config"] = processor_workflow.config  # 添加config到状态中
        decision = processor_workflow.should_collapse(after_collect)
        assert decision == "human_review"

        # 5. 执行人工审核
        review_result = await processor_workflow.human_review(after_collect)
        assert review_result["metadata"]["human_review_status"] == "completed"
        assert review_result["metadata"]["review_approved"] is True

        # 6. 检查后续流程
        updated_state = {**after_collect, **review_result}
        next_decision = processor_workflow.should_collapse(updated_state)
        assert next_decision == "generate_final_summary"
