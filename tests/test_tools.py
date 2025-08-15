import pytest
from fastmcp import Client
from pdf_query_mcp.server import mcp


@pytest.mark.asyncio
async def test_query_pdf_requires_api_key(monkeypatch):

	monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

	async with Client(mcp) as client:
		result = await client.call_tool(
			"query_pdf",
			{"source": "https://example.com/doc.pdf", "queries": ["What is this?"], "enable_grounding": False},
		)

		assert result.data["status"] == "error"
		assert "Google API key is required" in result.data["error"]


@pytest.mark.asyncio
async def test_summarize_pdf_requires_api_key(monkeypatch):

	monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

	async with Client(mcp) as client:
		result = await client.call_tool(
			"summarize_pdf",
			{"source": "https://example.com/doc.pdf", "summary_type": "brief", "enable_grounding": False},
		)

		assert result.data["status"] == "error"
		assert "Google API key is required" in result.data["error"]


@pytest.mark.asyncio
async def test_query_multiple_pdfs_requires_api_key(monkeypatch):

	monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

	async with Client(mcp) as client:
		result = await client.call_tool(
			"query_multiple_pdfs",
			{"sources": ["https://example.com/a.pdf", "https://example.com/b.pdf"], "queries": ["Compare"], "enable_grounding": False},
		)

		assert result.data["status"] == "error"
		assert "Google API key is required" in result.data["error"]


@pytest.mark.asyncio
async def test_compare_pdfs_requires_api_key(monkeypatch):

	monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

	async with Client(mcp) as client:
		result = await client.call_tool(
			"compare_pdfs",
			{"sources": ["https://example.com/a.pdf", "https://example.com/b.pdf"], "comparison_query": "What differs?", "enable_grounding": False},
		)

		assert result.data["status"] == "error"
		assert "Google API key is required" in result.data["error"]


@pytest.mark.asyncio
async def test_list_tools_in_memory():
	async with Client(mcp) as client:
		tools = await client.list_tools()
		names = [t.name for t in (tools.tools if hasattr(tools, "tools") else tools)]
		for expected in ["query_pdf", "query_multiple_pdfs", "compare_pdfs", "summarize_pdf"]:
			assert expected in names


