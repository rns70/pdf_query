import io
import os
import tempfile

import pytest

from pdf_query_mcp.server import _is_url, _get_pdf_size, _add_inline_citations, _process_grounding_metadata


def test_is_url_true_false_cases():

	assert _is_url("https://example.com/file.pdf") is True
	assert _is_url("http://example.com") is True
	assert _is_url("/absolute/path/to/file.pdf") is False
	assert _is_url("relative/path.pdf") is False
	assert _is_url("not a url") is False


def test_get_pdf_size_local_file():

	with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
		data = b"%PDF-1.4\n%EOF" * 10
		tmp.write(data)
		tmp_path = tmp.name

	try:
		size = _get_pdf_size(tmp_path)
		assert size == len(data)
	finally:
		os.remove(tmp_path)


def test_get_pdf_size_url(monkeypatch):

	class DummyResponse:
		def __init__(self, headers):
			self.headers = headers

	def fake_head(url, timeout):
		assert url == "https://example.com/file.pdf"
		assert timeout == 30
		return DummyResponse({"content-length": "12345"})

	import httpx

	monkeypatch.setattr(httpx, "head", fake_head)

	size = _get_pdf_size("https://example.com/file.pdf")
	assert size == 12345


def test_add_inline_citations_inserts_links():

	text = "This is some answer text."
	# Insert after last character (end_index=len(text))
	grounding_info = {
		"grounded": True,
		"citations": [
			{"text": "answer text", "start_index": 13, "end_index": len(text), "source_indices": [0]},
		],
		"sources": [
			{"index": 1, "title": "Src", "uri": "https://src"},
		],
	}

	with_cites = _add_inline_citations(text, grounding_info)
	assert with_cites.endswith(" [1](https://src)")


def test_process_grounding_metadata_happy_path():

	class DummyWeb:
		def __init__(self, title, uri):
			self.title = title
			self.uri = uri

	class DummyChunk:
		def __init__(self, web):
			self.web = web

	class DummySegment:
		def __init__(self, text, start_index, end_index):
			self.text = text
			self.start_index = start_index
			self.end_index = end_index

	class DummySupport:
		def __init__(self, segment, grounding_chunk_indices):
			self.segment = segment
			self.grounding_chunk_indices = grounding_chunk_indices

	class DummyMetadata:
		def __init__(self):
			self.web_search_queries = ["query one", "query two"]
			self.grounding_chunks = [
				DummyChunk(DummyWeb("Title A", "https://a")),
				DummyChunk(DummyWeb("Title B", "https://b")),
			]
			self.grounding_supports = [
				DummySupport(DummySegment("snippet", 0, 7), [0]),
			]

	class DummyCandidate:
		def __init__(self):
			self.grounding_metadata = DummyMetadata()

	class DummyResponse:
		def __init__(self):
			self.candidates = [DummyCandidate()]

	resp = _process_grounding_metadata(DummyResponse())

	assert resp["grounded"] is True
	assert resp["search_queries"] == ["query one", "query two"]
	assert len(resp["sources"]) == 2
	assert resp["sources"][0]["index"] == 1
	assert resp["sources"][0]["uri"] == "https://a"
	assert len(resp["citations"]) == 1
	assert resp["citations"][0]["text"] == "snippet"
	assert resp["citations"][0]["source_indices"] == [0]


