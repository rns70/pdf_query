"""PDF Query MCP Server using Google Gemini API with native PDF understanding."""

import os
import io
import pathlib
from typing import Optional, List, Union
from urllib.parse import urlparse

import httpx
from google import genai
from google.genai import types
from fastmcp import FastMCP, Context


def _is_url(source: str) -> bool:
    """Check if source is a URL."""
    try:
        result = urlparse(source)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def _get_pdf_size(source: str) -> int:
    """Get PDF size in bytes."""
    if _is_url(source):
        response = httpx.head(source, timeout=30)
        return int(response.headers.get("content-length", 0))
    else:
        return pathlib.Path(source).stat().st_size


def _create_pdf_part(
    source: str, use_file_api: bool = False, client: Optional[genai.Client] = None
) -> Union[types.Part, types.File]:
    """Create a PDF part for Gemini API from source."""
    if _is_url(source):
        # Download PDF from URL
        doc_data = httpx.get(source, timeout=60).content

        if use_file_api and client:
            # Use File API for large PDFs
            doc_io = io.BytesIO(doc_data)
            return client.files.upload(
                file=doc_io, config=dict(mime_type="application/pdf")
            )
        else:
            # Use inline data for smaller PDFs
            return types.Part.from_bytes(data=doc_data, mime_type="application/pdf")
    else:
        # Local file
        filepath = pathlib.Path(source)
        if not filepath.exists():
            raise FileNotFoundError(f"PDF file not found: {source}")

        if use_file_api and client:
            # Use File API for large PDFs
            return client.files.upload(file=filepath)
        else:
            # Use inline data for smaller PDFs
            return types.Part.from_bytes(
                data=filepath.read_bytes(), mime_type="application/pdf"
            )


def _process_grounding_metadata(response) -> dict:
    """Process grounding metadata from Gemini response to extract citations and search info."""
    grounding_info = {}
    
    try:
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                metadata = candidate.grounding_metadata
                
                # Extract search queries
                if hasattr(metadata, 'web_search_queries'):
                    grounding_info['search_queries'] = list(metadata.web_search_queries)
                
                # Extract grounding chunks (sources)
                if hasattr(metadata, 'grounding_chunks'):
                    sources = []
                    for i, chunk in enumerate(metadata.grounding_chunks):
                        if hasattr(chunk, 'web') and chunk.web:
                            sources.append({
                                'index': i + 1,
                                'title': chunk.web.title if hasattr(chunk.web, 'title') else 'Unknown',
                                'uri': chunk.web.uri if hasattr(chunk.web, 'uri') else ''
                            })
                    grounding_info['sources'] = sources
                
                # Extract grounding supports (citations)
                if hasattr(metadata, 'grounding_supports'):
                    citations = []
                    for support in metadata.grounding_supports:
                        if hasattr(support, 'segment') and hasattr(support, 'grounding_chunk_indices'):
                            citations.append({
                                'text': support.segment.text if hasattr(support.segment, 'text') else '',
                                'start_index': support.segment.start_index if hasattr(support.segment, 'start_index') else 0,
                                'end_index': support.segment.end_index if hasattr(support.segment, 'end_index') else 0,
                                'source_indices': list(support.grounding_chunk_indices)
                            })
                    grounding_info['citations'] = citations
                
                grounding_info['grounded'] = True
            else:
                grounding_info['grounded'] = False
        else:
            grounding_info['grounded'] = False
            
    except Exception as e:
        grounding_info['grounded'] = False
        grounding_info['error'] = str(e)
    
    return grounding_info


def _add_inline_citations(text: str, grounding_info: dict) -> str:
    """Add inline citations to text based on grounding metadata."""
    if not grounding_info.get('grounded') or not grounding_info.get('citations') or not grounding_info.get('sources'):
        return text
    
    try:
        citations = grounding_info['citations']
        sources = grounding_info['sources']
        
        # Sort citations by end_index in descending order to avoid shifting issues
        sorted_citations = sorted(citations, key=lambda c: c['end_index'], reverse=True)
        
        for citation in sorted_citations:
            end_index = citation['end_index']
            source_indices = citation['source_indices']
            
            if source_indices:
                # Create citation links
                citation_links = []
                for idx in source_indices:
                    # Find source by index (adjusting for 0-based vs 1-based indexing)
                    matching_sources = [s for s in sources if s['index'] == idx + 1]
                    if matching_sources:
                        source = matching_sources[0]
                        citation_links.append(f"[{source['index']}]({source['uri']})")
                
                if citation_links:
                    citation_string = " " + ", ".join(citation_links)
                    text = text[:end_index] + citation_string + text[end_index:]
        
        return text
    except Exception:
        # If citation processing fails, return original text
        return text


# Initialize FastMCP server
mcp = FastMCP("PDF Query Server")


@mcp.tool
async def query_pdf(
    source: str,
    queries: List[str],
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-pro",
    expected_number_of_words: Optional[int] = None,
    enable_grounding: bool = False,
    ctx: Context = None,
) -> dict:
    """
    Query a single PDF document using Google Gemini's native PDF understanding.

    Args:
        source: PDF file path (must be absolute path) or URL
        queries: List of questions to ask about the PDF. The model will be called once per query with context caching.
        api_key: Google GenAI API key (optional if set in GOOGLE_API_KEY environment variable)
        model: Gemini model to use (default: gemini-2.5-pro). You can switch to
               'gemini-2.5-flash' if you encounter rate limits or want lower latency.
        expected_number_of_words: Expected length of response in words (optional, helps optimize response length)
        enable_grounding: Enable Google Search grounding for real-time information and citations (default: false)

    Returns:
        Dictionary containing the answer and source information
    """
    if ctx:
        await ctx.info(f"Starting PDF query for source: {source}")

    try:
        # Get API key
        actual_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not actual_api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable or provide api_key parameter."
            )

        # Initialize client
        client = genai.Client(api_key=actual_api_key)

        # Check PDF size to determine if we should use File API
        pdf_size = _get_pdf_size(source)
        use_file_api = pdf_size > 20_000_000  # 20MB threshold

        if ctx:
            size_mb = pdf_size / (1024 * 1024)
            await ctx.info(
                f"PDF size: {size_mb:.1f}MB - Using {'File API' if use_file_api else 'inline data'}"
            )

        # Create PDF part
        if ctx:
            await ctx.info("Processing PDF...")
        pdf_part = _create_pdf_part(source, use_file_api, client)

        # Create a cached content for the PDF to enable efficient repeated queries
        cached_name = None
        try:
            if ctx:
                await ctx.info("Creating cached content for PDF context...")
            cache_config = types.CreateCachedContentConfig(
                contents=[types.Content(role="user", parts=[pdf_part])]
            )
            cached = client.caches.create(model=model, config=cache_config)
            cached_name = cached.name
        except Exception as e:
            # Fallback when content is too small for caching or cache creation fails
            if ctx:
                await ctx.info(f"Cache creation skipped: {e}")

        # Configure grounding and caching if available
        base_config = None
        if enable_grounding:
            if ctx:
                await ctx.info("Enabling Google Search grounding...")
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            base_config = types.GenerateContentConfig(
                tools=[grounding_tool], max_output_tokens=65535,
                **({"cached_content": cached_name} if cached_name else {})
            )
        else:
            base_config = types.GenerateContentConfig(
                max_output_tokens=65535,
                **({"cached_content": cached_name} if cached_name else {})
            )

        source_info = (
            f"PDF from URL: {source}" if _is_url(source) else f"Local PDF: {source}"
        )

        # Iterate over queries using context caching
        answers: List[str] = []
        groundings: List[dict] = []

        if ctx:
            await ctx.info(f"Running {len(queries)} queries with cached context...")

        for q in queries:
            final_query = q
            if expected_number_of_words:
                final_query = f"{q}\n\nPlease provide a response of approximately {expected_number_of_words} words."

            if cached_name:
                # Use cached context; only send the query
                response = client.models.generate_content(
                    model=model, contents=[final_query], config=base_config
                )
            else:
                # No cache; include the PDF part each time
                response = client.models.generate_content(
                    model=model, contents=[pdf_part, final_query], config=base_config
                )

            if enable_grounding:
                g_info = _process_grounding_metadata(response)
                groundings.append(g_info)
                text = response.text
                if g_info.get('grounded'):
                    text = _add_inline_citations(text, g_info)
                answers.append(text)
            else:
                answers.append(response.text)

        if ctx:
            await ctx.info("Queries completed successfully")

        result = {
            "answers": answers,
            "source_info": source_info,
            "queries": queries,
            "pdf_size_mb": round(pdf_size / (1024 * 1024), 2),
            "model_used": model,
            "processing_method": "File API" if use_file_api else "Inline data",
            "grounding_enabled": enable_grounding,
            "cache_used": bool(cached_name),
            "status": "success",
        }

        if enable_grounding:
            result["grounding_per_query"] = groundings

        return result

    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to query PDF: {e}")
        return {"error": str(e), "status": "error"}


@mcp.tool
async def query_multiple_pdfs(
    sources: List[str],
    queries: List[str],
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-pro",
    expected_number_of_words: Optional[int] = None,
    enable_grounding: bool = False,
    ctx: Context = None,
) -> dict:
    """
    Query multiple PDF documents at once using Google Gemini's native PDF understanding.

    Args:
        sources: List of PDF file paths (must be absolute paths) or URLs
        queries: List of questions to ask about the PDFs. The model will be called once per query with cached context of all PDFs.
        api_key: Google GenAI API key (optional if set in GOOGLE_API_KEY environment variable)
        model: Gemini model to use (default: gemini-2.5-pro). You can switch to
               'gemini-2.5-flash' if you encounter rate limits or want lower latency.
        expected_number_of_words: Expected length of response in words (optional, helps optimize response length)
        enable_grounding: Enable Google Search grounding for real-time information and citations (default: false)

    Returns:
        Dictionary containing the answer and source information
    """
    if ctx:
        await ctx.info(f"Starting multi-PDF query for {len(sources)} documents")

    try:
        # Get API key
        actual_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not actual_api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable or provide api_key parameter."
            )

        # Initialize client
        client = genai.Client(api_key=actual_api_key)

        # Process all PDFs
        pdf_parts = []
        source_info = []
        total_size = 0

        for i, source in enumerate(sources):
            if ctx:
                await ctx.info(f"Processing PDF {i + 1}/{len(sources)}: {source}")

            pdf_size = _get_pdf_size(source)
            total_size += pdf_size

            # For multiple PDFs, prefer File API for better handling
            use_file_api = pdf_size > 10_000_000  # Lower threshold for multiple PDFs

            pdf_part = _create_pdf_part(source, use_file_api, client)
            pdf_parts.append(pdf_part)

            source_type = "URL" if _is_url(source) else "Local file"
            source_info.append(
                f"{source_type}: {source} ({pdf_size / (1024 * 1024):.1f}MB)"
            )

        # Create a single cached content for all PDFs
        cached_name = None
        try:
            if ctx:
                await ctx.info("Creating cached content for multi-PDF context...")
            cache_config = types.CreateCachedContentConfig(
                contents=[types.Content(role="user", parts=pdf_parts)]
            )
            cached = client.caches.create(model=model, config=cache_config)
            cached_name = cached.name
        except Exception as e:
            if ctx:
                await ctx.info(f"Cache creation skipped: {e}")

        # Configure grounding if enabled
        base_config = None
        if enable_grounding:
            if ctx:
                await ctx.info("Enabling Google Search grounding...")
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            base_config = types.GenerateContentConfig(
                tools=[grounding_tool], max_output_tokens=65535,
                **({"cached_content": cached_name} if cached_name else {})
            )
        else:
            base_config = types.GenerateContentConfig(
                max_output_tokens=65535,
                **({"cached_content": cached_name} if cached_name else {})
            )

        # Iterate over queries using cached context
        answers: List[str] = []
        groundings: List[dict] = []

        if ctx:
            await ctx.info(f"Running {len(queries)} queries with cached context...")

        for q in queries:
            final_query = q
            if expected_number_of_words:
                final_query = f"{q}\n\nPlease provide a response of approximately {expected_number_of_words} words."

            if cached_name:
                response = client.models.generate_content(
                    model=model, contents=[final_query], config=base_config
                )
            else:
                response = client.models.generate_content(
                    model=model, contents=pdf_parts + [final_query], config=base_config
                )

            if enable_grounding:
                g_info = _process_grounding_metadata(response)
                groundings.append(g_info)
                text = response.text
                if g_info.get('grounded'):
                    text = _add_inline_citations(text, g_info)
                answers.append(text)
            else:
                answers.append(response.text)

        if ctx:
            await ctx.info("Multi-PDF queries completed successfully")

        result = {
            "answers": answers,
            "sources": source_info,
            "queries": queries,
            "total_pdfs": len(sources),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "model_used": model,
            "grounding_enabled": enable_grounding,
            "cache_used": bool(cached_name),
            "status": "success",
        }

        if enable_grounding:
            result["grounding_per_query"] = groundings

        return result

    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to query multiple PDFs: {e}")
        return {"error": str(e), "status": "error"}


@mcp.tool
async def compare_pdfs(
    sources: List[str],
    comparison_query: str,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-pro",
    expected_number_of_words: Optional[int] = None,
    enable_grounding: bool = False,
    ctx: Context = None,
) -> dict:
    """
    Compare multiple PDF documents using Google Gemini's native PDF understanding.

    Args:
        sources: List of PDF file paths (must be absolute paths) or URLs to compare
        comparison_query: Specific comparison question (e.g., "What are the differences between these documents?")
        api_key: Google GenAI API key (optional if set in GOOGLE_API_KEY environment variable)
        model: Gemini model to use (default: gemini-2.5-pro). You can switch to
               'gemini-2.5-flash' if you encounter rate limits or want lower latency.
        expected_number_of_words: Expected length of response in words (optional, helps optimize response length)
        enable_grounding: Enable Google Search grounding for real-time information and citations (default: false)

    Returns:
        Dictionary containing the comparison results
    """
    if ctx:
        await ctx.info(f"Starting PDF comparison for {len(sources)} documents")

    try:
        # Get API key
        actual_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not actual_api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable or provide api_key parameter."
            )

        # Initialize client
        client = genai.Client(api_key=actual_api_key)

        # Process all PDFs
        pdf_parts = []
        source_info = []
        total_size = 0

        for i, source in enumerate(sources):
            if ctx:
                await ctx.info(f"Processing PDF {i + 1}/{len(sources)}: {source}")

            pdf_size = _get_pdf_size(source)
            total_size += pdf_size

            # For comparison, prefer File API for better handling
            use_file_api = pdf_size > 10_000_000  # Lower threshold for comparison

            pdf_part = _create_pdf_part(source, use_file_api, client)
            pdf_parts.append(pdf_part)

            source_type = "URL" if _is_url(source) else "Local file"
            source_info.append(
                f"Document {i + 1}: {source_type} - {source} ({pdf_size / (1024 * 1024):.1f}MB)"
            )

        # Create comparison prompt
        word_count_guidance = f"\n\nPlease provide a response of approximately {expected_number_of_words} words." if expected_number_of_words else ""
        
        full_prompt = f"""Please compare the following {len(sources)} PDF documents and answer this question: {comparison_query}

When comparing, please:
1. Identify the key similarities and differences
2. Highlight unique aspects of each document
3. Provide specific examples from each document
4. Organize your response clearly

Question: {comparison_query}{word_count_guidance}"""

        # Query Gemini with all PDFs
        if ctx:
            await ctx.info("Performing comparison with Gemini...")

        # Configure grounding if enabled
        config = None
        if enable_grounding:
            if ctx:
                await ctx.info("Enabling Google Search grounding...")
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(tools=[grounding_tool], max_output_tokens=65535)

        contents = pdf_parts + [full_prompt]
        response = client.models.generate_content(model=model, contents=contents, config=config)

        # Process grounding metadata if grounding was enabled
        grounding_info = {}
        if enable_grounding:
            if ctx:
                await ctx.info("Processing grounding metadata...")
            grounding_info = _process_grounding_metadata(response)
            
            # Add inline citations to the comparison result if grounded
            comparison_text = response.text
            if grounding_info.get('grounded'):
                comparison_text = _add_inline_citations(response.text, grounding_info)
        else:
            comparison_text = response.text

        if ctx:
            await ctx.info("PDF comparison completed successfully")

        result = {
            "comparison_result": comparison_text,
            "sources": source_info,
            "total_pdfs": len(sources),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "comparison_query": comparison_query,
            "model_used": model,
            "grounding_enabled": enable_grounding,
            "status": "success",
        }

        # Add grounding information if available
        if enable_grounding and grounding_info:
            result["grounding"] = grounding_info

        return result

    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to compare PDFs: {e}")
        return {"error": str(e), "status": "error"}


@mcp.tool
async def summarize_pdf(
    source: str,
    summary_type: str = "comprehensive",
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-pro",
    expected_number_of_words: Optional[int] = None,
    enable_grounding: bool = False,
    ctx: Context = None,
) -> dict:
    """
    Summarize a PDF document using Google Gemini's native PDF understanding.

    Args:
        source: PDF file path (must be absolute path) or URL
        summary_type: Type of summary ('brief', 'comprehensive', 'technical', 'executive')
        api_key: Google GenAI API key (optional if set in GOOGLE_API_KEY environment variable)
        model: Gemini model to use (default: gemini-2.5-pro). You can switch to
               'gemini-2.5-flash' if you encounter rate limits or want lower latency.
        expected_number_of_words: Expected length of response in words (optional, helps optimize response length)
        enable_grounding: Enable Google Search grounding for real-time information and citations (default: false)

    Returns:
        Dictionary containing the summary and source information
    """
    if ctx:
        await ctx.info(f"Starting PDF summarization for: {source}")

    try:
        # Get API key
        actual_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not actual_api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable or provide api_key parameter."
            )

        # Initialize client
        client = genai.Client(api_key=actual_api_key)

        # Check PDF size
        pdf_size = _get_pdf_size(source)
        use_file_api = pdf_size > 20_000_000  # 20MB threshold

        if ctx:
            size_mb = pdf_size / (1024 * 1024)
            await ctx.info(
                f"PDF size: {size_mb:.1f}MB - Using {'File API' if use_file_api else 'inline data'}"
            )

        # Create PDF part
        if ctx:
            await ctx.info("Processing PDF...")
        pdf_part = _create_pdf_part(source, use_file_api, client)

        # Create summary prompt based on type
        summary_prompts = {
            "brief": "Provide a brief summary of this document in 2-3 paragraphs.",
            "comprehensive": "Provide a comprehensive summary of this document, including key points, main arguments, conclusions, and important details.",
            "technical": "Provide a technical summary focusing on methodologies, data, findings, and technical details.",
            "executive": "Provide an executive summary suitable for leadership, focusing on key findings, recommendations, and business implications.",
        }

        base_prompt = summary_prompts.get(summary_type, summary_prompts["comprehensive"])
        
        # Add word count guidance if provided
        prompt = base_prompt
        if expected_number_of_words:
            prompt = f"{base_prompt} Please provide a response of approximately {expected_number_of_words} words."

        # Query Gemini
        if ctx:
            await ctx.info(f"Generating {summary_type} summary with Gemini...")

        # Configure grounding if enabled
        config = None
        if enable_grounding:
            if ctx:
                await ctx.info("Enabling Google Search grounding...")
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(tools=[grounding_tool], max_output_tokens=65535)

        response = client.models.generate_content(
            model=model, contents=[pdf_part, prompt], config=config
        )

        source_info = (
            f"PDF from URL: {source}" if _is_url(source) else f"Local PDF: {source}"
        )

        # Process grounding metadata if grounding was enabled
        grounding_info = {}
        if enable_grounding:
            if ctx:
                await ctx.info("Processing grounding metadata...")
            grounding_info = _process_grounding_metadata(response)
            
            # Add inline citations to the summary if grounded
            summary_text = response.text
            if grounding_info.get('grounded'):
                summary_text = _add_inline_citations(response.text, grounding_info)
        else:
            summary_text = response.text

        if ctx:
            await ctx.info("Summarization completed successfully")

        result = {
            "summary": summary_text,
            "source_info": source_info,
            "summary_type": summary_type,
            "pdf_size_mb": round(pdf_size / (1024 * 1024), 2),
            "model_used": model,
            "grounding_enabled": enable_grounding,
            "status": "success",
        }

        # Add grounding information if available
        if enable_grounding and grounding_info:
            result["grounding"] = grounding_info

        return result

    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to summarize PDF: {e}")
        return {"error": str(e), "status": "error"}


def main():
    """Main entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
