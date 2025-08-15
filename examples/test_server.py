#!/usr/bin/env python3
"""
Example script to test the PDF Query MCP Server using FastMCP testing patterns.

This script demonstrates how to properly test the PDF Query MCP server
using the FastMCP Client for in-memory testing.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path so we can import our module
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import Client
from pdf_query_mcp.server import mcp


async def test_pdf_summarization():
    """Test summarizing a PDF using the MCP client."""
    print("Testing PDF summarization...")
    
    # You can use either a local file or URL
    pdf_source = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    try:
        # Use FastMCP Client for in-memory testing
        async with Client(mcp) as client:
            result = await client.call_tool("summarize_pdf", {
                "source": pdf_source,
                "summary_type": "brief"
            })
            
            if hasattr(result, 'data') and result.data["status"] == "success":
                print(f"✅ Source: {result.data['source_info']}")
                print(f"✅ Summary type: {result.data['summary_type']}")
                print(f"✅ PDF size: {result.data['pdf_size_mb']}MB")
                print(f"✅ Model used: {result.data['model_used']}")
                print(f"✅ Summary preview: {result.data['summary'][:200]}...")
                return True
            else:
                error_msg = result.data.get("error", "Unknown error") if hasattr(result, 'data') else str(result)
                print(f"❌ Error: {error_msg}")
                return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_url_pdf_query():
    """Test querying a PDF from a URL using the MCP client."""
    print("Testing URL PDF query...")
    
    # Example URL - replace with a real PDF URL
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    query = "What does this document contain?"
    
    try:
        # Use FastMCP Client for in-memory testing
        async with Client(mcp) as client:
            result = await client.call_tool("query_pdf", {
                "source": pdf_url,
                "query": query
            })
            
            if hasattr(result, 'data') and result.data["status"] == "success":
                print(f"✅ Source: {result.data['source_info']}")
                print(f"✅ Answer: {result.data['answer']}")
                return True
            else:
                error_msg = result.data.get("error", "Unknown error") if hasattr(result, 'data') else str(result)
                print(f"❌ Error: {error_msg}")
                return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_local_pdf_query():
    """Test querying a local PDF file using the MCP client."""
    print("Testing local PDF query...")
    
    # You'll need to replace this with an actual PDF file path
    pdf_path = "example.pdf"
    query = "What is this document about?"
    
    try:
        # Use FastMCP Client for in-memory testing
        async with Client(mcp) as client:
            result = await client.call_tool("query_pdf", {
                "source": pdf_path,
                "query": query
            })
            
            if hasattr(result, 'data') and result.data["status"] == "success":
                print(f"✅ Source: {result.data['source_info']}")
                print(f"✅ Answer: {result.data['answer']}")
                return True
            else:
                error_msg = result.data.get("error", "Unknown error") if hasattr(result, 'data') else str(result)
                print(f"❌ Error: {error_msg}")
                return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_server_info():
    """Test basic server information."""
    print("Testing server info...")
    
    try:
        async with Client(mcp) as client:
            # List available tools
            tools = await client.list_tools()
            # Handle both list and object with tools attribute
            if hasattr(tools, 'tools'):
                tool_names = [tool.name for tool in tools.tools]
            else:
                tool_names = [tool.name for tool in tools]
            print(f"✅ Available tools: {tool_names}")
            
            expected_tools = ["query_pdf", "query_multiple_pdfs", "compare_pdfs", "summarize_pdf"]
            for tool in expected_tools:
                if tool in tool_names:
                    print(f"✅ Tool '{tool}' is available")
                else:
                    print(f"❌ Tool '{tool}' is missing")
                    return False
            
            return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_multiple_pdfs():
    """Test querying multiple PDFs at once."""
    print("Testing multiple PDF query...")
    
    # Use two sample PDFs
    pdf_sources = [
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"  # Using same for demo
    ]
    
    try:
        async with Client(mcp) as client:
            result = await client.call_tool("query_multiple_pdfs", {
                "sources": pdf_sources,
                "query": "What do these documents contain?"
            })
            
            if hasattr(result, 'data') and result.data["status"] == "success":
                print(f"✅ Total PDFs: {result.data['total_pdfs']}")
                print(f"✅ Total size: {result.data['total_size_mb']}MB")
                print(f"✅ Model used: {result.data['model_used']}")
                print(f"✅ Answer preview: {result.data['answer'][:200]}...")
                return True
            else:
                error_msg = result.data.get("error", "Unknown error") if hasattr(result, 'data') else str(result)
                print(f"❌ Error: {error_msg}")
                return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Main test function."""
    print("PDF Query MCP Server Test Script (Using FastMCP Client)")
    print("=" * 60)
    
    # Check if GOOGLE_API_KEY is set
    api_key_available = bool(os.getenv("GOOGLE_API_KEY"))
    if not api_key_available:
        print("⚠️  Warning: GOOGLE_API_KEY environment variable not set!")
        print("   Some tests that require AI queries will be skipped.")
        print()
    
    # Test results
    results = {}
    
    # Test server info (always works)
    print("1. Server Information")
    results["server_info"] = await test_server_info()
    print()
    
    # Test PDF summarization (requires API key but works without it for testing)
    print("2. PDF Summarization")
    results["pdf_summarization"] = await test_pdf_summarization()
    print()
    
    # Test PDF querying (requires API key)
    if api_key_available:
        print("3. PDF Query from URL")
        results["url_query"] = await test_url_pdf_query()
        print()
        
        print("4. Multiple PDF Query")
        results["multiple_pdfs"] = await test_multiple_pdfs()
        print()
        
        # Uncomment to test local PDF (make sure to have a PDF file)
        # print("5. PDF Query from Local File")
        # results["local_query"] = await test_local_pdf_query()
        # print()
    else:
        print("3. PDF Query Tests - SKIPPED (no API key)")
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary:")
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 