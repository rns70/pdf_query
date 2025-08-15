# PDF Query MCP Server

A Model Context Protocol (MCP) server for querying PDF documents using Google Gemini's native PDF understanding capabilities. This server supports both local PDF files and PDFs from URLs, with advanced features for multi-document analysis.

## Features

- **Native PDF Understanding**: Uses Google Gemini's native PDF processing (not just text extraction)
- **Multi-Document Support**: Query multiple PDFs simultaneously (up to 1000 pages total)
- **Document Comparison**: Compare multiple PDFs side-by-side
- **Smart Summarization**: Generate different types of summaries (brief, comprehensive, technical, executive)
- **Google Search Grounding**: Optional real-time web search integration with automatic citations
- **Intelligent File Handling**: Automatically chooses between inline data and File API based on document size
- **Support for Complex Content**: Handles text, images, diagrams, charts, and tables
- **Local and URL Support**: Works with both local files and URLs
- **Built with FastMCP v2**: Easy integration with MCP-compatible clients

## Requirements

- Python 3.10+
- Google Gemini API key
- uv (recommended) or pip for dependency management

## Installation

1. Clone or download this project
2. Install dependencies using uv:

```bash
uv pip install -e .
```

Or with pip:

```bash
pip install -e .
```

## Setup

1. Get a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Set the API key as an environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Alternatively, you can provide the API key directly when calling the tools.

## Usage

### Running the Server

Start the MCP server:

```bash
pdf-query-mcp
```

Or run directly:

```bash
python -m pdf_query_mcp.server
```

### Available Tools

#### 1. `query_pdf`

Query a single PDF document using Google Gemini's native PDF understanding.

**Parameters:**
- `source` (str): PDF file path (must be absolute path) or URL
- `queries` (list[str]): Questions to ask about the PDF. The server will reuse cached PDF context and run one LLM call per query.
- `api_key` (str, optional): Google Gemini API key
- `model` (str, optional): Gemini model to use (default: gemini-2.5-pro). You can switch to `gemini-2.5-flash` if you encounter rate limits or want lower latency.
- `expected_number_of_words` (int, optional): Expected length of response in words
- `enable_grounding` (bool, optional): Enable Google Search grounding for real-time information and citations (default: false)

**Example:**
```json
{
  "source": "https://example.com/document.pdf",
  "queries": ["What is the main topic of this document?"],
  "model": "gemini-2.5-pro",
  "expected_number_of_words": 150,
  "enable_grounding": false
}
```

#### 2. `query_multiple_pdfs`

Query multiple PDF documents simultaneously.

**Parameters:**
- `sources` (list[str]): List of PDF file paths (must be absolute paths) or URLs
- `queries` (list[str]): Questions to ask about the PDFs. The server will cache combined PDF context and run one LLM call per query.
- `api_key` (str, optional): Google Gemini API key
- `model` (str, optional): Gemini model to use (default: gemini-2.5-pro). You can switch to `gemini-2.5-flash` if you encounter rate limits or want lower latency.
- `expected_number_of_words` (int, optional): Expected length of response in words
- `enable_grounding` (bool, optional): Enable Google Search grounding for real-time information and citations (default: false)

**Example:**
```json
{
  "sources": ["https://example.com/doc1.pdf", "https://example.com/doc2.pdf"],
  "queries": ["What are the common themes across these documents?"],
  "expected_number_of_words": 300,
  "enable_grounding": false
}
```

#### 3. `compare_pdfs`

Compare multiple PDF documents side-by-side.

**Parameters:**
- `sources` (list[str]): List of PDF file paths (must be absolute paths) or URLs to compare
- `comparison_query` (str): Specific comparison question
- `api_key` (str, optional): Google Gemini API key
- `model` (str, optional): Gemini model to use (default: gemini-2.5-pro). You can switch to `gemini-2.5-flash` if you encounter rate limits or want lower latency.
- `expected_number_of_words` (int, optional): Expected length of response in words
- `enable_grounding` (bool, optional): Enable Google Search grounding for real-time information and citations (default: false)

**Example:**
```json
{
  "sources": ["https://example.com/report1.pdf", "https://example.com/report2.pdf"],
  "comparison_query": "What are the key differences between these two reports?",
  "expected_number_of_words": 400,
  "enable_grounding": false
}
```

#### 4. `summarize_pdf`

Generate different types of summaries for PDF documents.

**Parameters:**
- `source` (str): PDF file path (must be absolute path) or URL
- `summary_type` (str): Type of summary ('brief', 'comprehensive', 'technical', 'executive')
- `api_key` (str, optional): Google Gemini API key
- `model` (str, optional): Gemini model to use (default: gemini-2.5-pro). You can switch to `gemini-2.5-flash` if you encounter rate limits or want lower latency.
- `expected_number_of_words` (int, optional): Expected length of response in words
- `enable_grounding` (bool, optional): Enable Google Search grounding for real-time information and citations (default: false)

**Example:**
```json
{
  "source": "https://example.com/research.pdf",
  "summary_type": "executive",
  "expected_number_of_words": 200,
  "enable_grounding": false
}
```

### Example Usage with MCP Client

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "pdf_query_mcp.server"],
        env={"GOOGLE_API_KEY": "your-api-key-here"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # Query a PDF
            result = await session.call_tool(
                "query_pdf",
                {
                    "source": "https://example.com/document.pdf",
                    "query": "What are the key findings?"
                }
            )
            
            print(result.content)

asyncio.run(main())
```

## Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Your Google GenAI API key (required)

### Supported PDF Sources

- **Local files**: `/path/to/document.pdf`
- **URLs**: `https://example.com/document.pdf`

## Error Handling

The server includes comprehensive error handling for:
- Invalid PDF files
- Network issues when downloading from URLs
- API key validation
- Text extraction failures

All errors are returned in a structured format with status indicators.

## Development

### Setup Development Environment

```bash
# Install with development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Lint code
ruff check .
```

### Project Structure

```
pdf_query_mcp/
├── __init__.py
├── server.py          # Main MCP server implementation
├── pyproject.toml     # Project configuration
└── README.md         # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Troubleshooting

### Common Issues

1. **API Key not found**: Make sure `GOOGLE_API_KEY` is set in your environment
2. **PDF extraction fails**: Ensure the PDF is not password-protected and contains extractable text
3. **URL download fails**: Check network connectivity and ensure the URL is accessible

### Debug Mode

For debugging, you can run the server with additional logging:

```bash
PYTHONPATH=. python -m pdf_query_mcp.server --log-level DEBUG
``` 