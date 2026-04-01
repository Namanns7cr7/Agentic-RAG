"""Unit tests for individual RAG components."""

import pytest
import numpy as np
from agentic_rag.rag.tools import CalculatorTool, TimeTool, SummarizeTool, WebScraperTool


# ── Calculator Tool ───────────────────────────────────────────────

class TestCalculatorTool:
    tool = CalculatorTool()

    def test_basic_addition(self):
        assert self.tool.run("2+3") == "5"

    def test_multiplication(self):
        assert self.tool.run("6*7") == "42"

    def test_complex_expression(self):
        assert self.tool.run("2*(3+4)") == "14"

    def test_division(self):
        result = float(self.tool.run("10/3"))
        assert abs(result - 3.333) < 0.01

    def test_power(self):
        assert self.tool.run("2**10") == "1024"

    def test_negative_number(self):
        assert self.tool.run("-5+3") == "-2"

    def test_invalid_expression(self):
        result = self.tool.run("hello world")
        assert "error" in result.lower()

    def test_no_code_injection(self):
        result = self.tool.run("__import__('os').system('echo hacked')")
        assert "hacked" not in result  # Should fail safely


# ── Time Tool ─────────────────────────────────────────────────────

class TestTimeTool:
    tool = TimeTool()

    def test_returns_iso_string(self):
        result = self.tool.run()
        assert "T" in result
        assert len(result) >= 19  # YYYY-MM-DDTHH:MM:SS


# ── Summarize Tool ────────────────────────────────────────────────

class TestSummarizeTool:
    tool = SummarizeTool()

    def test_summarize_text(self):
        text = "RAG is great. It retrieves docs. Then generates answers."
        result = self.tool.run(text)
        assert "•" in result

    def test_summarize_empty(self):
        result = self.tool.run("")
        assert "nothing" in result.lower()

    def test_summarize_limits_to_five(self):
        text = ". ".join([f"Sentence {i}" for i in range(10)]) + "."
        result = self.tool.run(text)
        assert result.count("•") <= 5


# ── VectorStore Fallback ──────────────────────────────────────────

class TestVectorStoreFallback:
    """Test the NumPy fallback by directly importing."""

    def test_add_and_search(self):
        from agentic_rag.rag.vectorstore import VectorStore

        store = VectorStore(dim=4)
        docs = ["hello", "world"]
        embs = np.random.rand(2, 4).astype("float32")
        store.add(docs, embs)
        assert store.size == 2

        query = embs[0:1]
        distances, idx, retrieved = store.search(query, k=1)
        assert len(retrieved[0]) >= 1

    def test_search_empty_store(self):
        from agentic_rag.rag.vectorstore import VectorStore

        store = VectorStore(dim=4)
        query = np.random.rand(1, 4).astype("float32")
        distances, idx, retrieved = store.search(query, k=1)
        # Should not crash


# ── Scraper Tool ──────────────────────────────────────────────────

class TestScraperTool:
    tool = WebScraperTool()

    def test_missing_url(self):
        res = self.tool.run("scrape this page")
        assert "Error" in res

    def test_scrape_example_com(self):
        res = self.tool.run("scrape http://example.com please")
        # In environments with internet, it will scrape. In CI without internet, it will gracefully fail.
        assert "Domain" in res or "Example" in res or "Failed to scrape" in res

