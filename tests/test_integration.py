import os
import pytest
from fastmcp import Client
from pdf_query_mcp.server import mcp


requires_api_key = pytest.mark.skipif(
	not os.getenv("GOOGLE_API_KEY"),
	reason="GOOGLE_API_KEY not set; skipping live Gemini integration tests",
)


@pytest.mark.asyncio
@requires_api_key
async def test_integration_query_pdf_live():
	url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
	async with Client(mcp) as client:
		result = await client.call_tool(
			"query_pdf",
			{"source": url, "queries": ["What is this document about?"], "model": "gemini-2.5-flash", "enable_grounding": False},
		)
		assert result.data["status"] == "success"
		assert isinstance(result.data.get("answers"), list) and len(result.data["answers"]) == 1 and isinstance(result.data["answers"][0], str)
		assert result.data.get("model_used") in ("gemini-2.5-flash", "gemini-2.5-flash-exp")
		assert result.data.get("grounding_enabled") is False
		assert "PDF from URL:" in result.data.get("source_info", "")
		assert result.data.get("pdf_size_mb", 0) >= 0
		# Caching may be false for tiny PDFs; still assert key exists
		assert "cache_used" in result.data
		# Heuristic that output seems related to the PDF
		joined = " ".join(result.data["answers"]).lower()
		assert ("dummy" in joined) or ("pdf" in joined)


@pytest.mark.asyncio
@requires_api_key
async def test_integration_summarize_pdf_live():
	url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
	async with Client(mcp) as client:
		result = await client.call_tool(
			"summarize_pdf",
			{"source": url, "summary_type": "brief", "model": "gemini-2.5-flash", "enable_grounding": False},
		)
		assert result.data["status"] == "success"
		assert isinstance(result.data.get("summary"), str) and len(result.data["summary"]) > 0
		assert result.data.get("model_used") in ("gemini-2.5-flash", "gemini-2.5-flash-exp")
		assert result.data.get("grounding_enabled") is False
		assert "PDF from URL:" in result.data.get("source_info", "")


@pytest.mark.asyncio
@requires_api_key
async def test_integration_query_pdf_with_word_count_and_api_key_override():
	url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
	api_key = os.getenv("GOOGLE_API_KEY")
	async with Client(mcp) as client:
		result = await client.call_tool(
			"query_pdf",
			{
				"source": url,
				"queries": ["Give a short description."],
				"model": "gemini-2.5-flash",
				"expected_number_of_words": 30,
				"enable_grounding": False,
				"api_key": api_key,
			},
		)
		assert result.data["status"] == "success"
		assert isinstance(result.data.get("answers"), list) and len(result.data["answers"]) == 1 and isinstance(result.data["answers"][0], str)
		assert result.data.get("grounding_enabled") is False
		assert "cache_used" in result.data


@pytest.mark.asyncio
@requires_api_key
async def test_integration_query_multiple_pdfs_live():
	urls = [
		"https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
		"https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
	]
	async with Client(mcp) as client:
		result = await client.call_tool(
			"query_multiple_pdfs",
			{"sources": urls, "queries": ["What do these contain?"], "expected_number_of_words": 40, "enable_grounding": False},
		)
		assert result.data["status"] == "success"
		assert result.data.get("total_pdfs") == 2
		assert isinstance(result.data.get("answers"), list) and len(result.data["answers"]) == 1 and isinstance(result.data["answers"][0], str)
		assert result.data.get("grounding_enabled") is False
		sources = result.data.get("sources", [])
		assert isinstance(sources, list) and len(sources) == 2
		assert "cache_used" in result.data
		# Heuristic check that model actually read PDF content (dummy.pdf includes the word "Dummy")
		joined = " ".join(result.data["answers"]).lower()
		assert ("dummy" in joined) or ("pdf" in joined)


@pytest.mark.asyncio
@requires_api_key
async def test_integration_compare_pdfs_live():
	urls = [
		"https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
		"https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
	]
	async with Client(mcp) as client:
		result = await client.call_tool(
			"compare_pdfs",
			{"sources": urls, "comparison_query": "Compare these two docs", "expected_number_of_words": 40, "enable_grounding": False},
		)
		assert result.data["status"] == "success"
		assert isinstance(result.data.get("comparison_result"), str) and len(result.data["comparison_result"]) > 0
		assert result.data.get("grounding_enabled") is False


