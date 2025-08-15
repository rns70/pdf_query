#!/usr/bin/env python3
"""
Usage example for the PDF Query MCP Server.

This script demonstrates how to use the various tools available in the PDF Query MCP server.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path so we can import our module
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import Client
from pdf_query_mcp.server import mcp


async def example_single_pdf_query():
    """Example: Query a single PDF document."""
    print("🔍 Single PDF Query Example")
    print("-" * 40)
    
    async with Client(mcp) as client:
        result = await client.call_tool("query_pdf", {
            "source": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "query": "What is this document about?",
            "model": "gemini-2.5-flash"
        })
        
        if hasattr(result, 'data') and result.data.get("status") == "success":
            print(f"✅ Answer: {result.data['answer']}")
            print(f"📄 Source: {result.data['source_info']}")
            print(f"⚙️  Model: {result.data['model_used']}")
        else:
            print(f"❌ Error: {result.data.get('error', 'Unknown error')}")
    
    print()


async def example_multiple_pdf_query():
    """Example: Query multiple PDF documents."""
    print("📚 Multiple PDF Query Example")
    print("-" * 40)
    
    async with Client(mcp) as client:
        result = await client.call_tool("query_multiple_pdfs", {
            "sources": [
                "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
            ],
            "query": "What do these documents contain?",
            "model": "gemini-2.5-flash"
        })
        
        if hasattr(result, 'data') and result.data.get("status") == "success":
            print(f"✅ Answer: {result.data['answer']}")
            print(f"📊 Total PDFs: {result.data['total_pdfs']}")
            print(f"💾 Total Size: {result.data['total_size_mb']}MB")
            print(f"⚙️  Model: {result.data['model_used']}")
        else:
            print(f"❌ Error: {result.data.get('error', 'Unknown error')}")
    
    print()


async def example_pdf_comparison():
    """Example: Compare multiple PDF documents."""
    print("🔍 PDF Comparison Example")
    print("-" * 40)
    
    async with Client(mcp) as client:
        result = await client.call_tool("compare_pdfs", {
            "sources": [
                "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
            ],
            "comparison_query": "What are the similarities and differences between these documents?",
            "model": "gemini-2.5-flash"
        })
        
        if hasattr(result, 'data') and result.data.get("status") == "success":
            print(f"✅ Comparison Result: {result.data['comparison_result']}")
            print(f"📊 Total PDFs: {result.data['total_pdfs']}")
            print(f"💾 Total Size: {result.data['total_size_mb']}MB")
            print(f"⚙️  Model: {result.data['model_used']}")
        else:
            print(f"❌ Error: {result.data.get('error', 'Unknown error')}")
    
    print()


async def example_pdf_summarization():
    """Example: Summarize a PDF document."""
    print("📝 PDF Summarization Example")
    print("-" * 40)
    
    async with Client(mcp) as client:
        result = await client.call_tool("summarize_pdf", {
            "source": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "summary_type": "brief",
            "model": "gemini-2.5-flash"
        })
        
        if hasattr(result, 'data') and result.data.get("status") == "success":
            print(f"✅ Summary: {result.data['summary']}")
            print(f"📄 Source: {result.data['source_info']}")
            print(f"📊 Summary Type: {result.data['summary_type']}")
            print(f"💾 PDF Size: {result.data['pdf_size_mb']}MB")
            print(f"⚙️  Model: {result.data['model_used']}")
        else:
            print(f"❌ Error: {result.data.get('error', 'Unknown error')}")
    
    print()


async def main():
    """Main example function."""
    print("🚀 PDF Query MCP Server Usage Examples")
    print("=" * 60)
    
    # Check if API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY environment variable not set!")
        print("   These examples will show error messages without a valid API key.")
        print("   Set your Google Gemini API key to see actual results.")
        print()
    
    # Run examples
    await example_single_pdf_query()
    await example_multiple_pdf_query()
    await example_pdf_comparison()
    await example_pdf_summarization()
    
    print("✨ Examples completed!")
    print("\nTo use with a real API key:")
    print("export GOOGLE_API_KEY='your-gemini-api-key-here'")
    print("python examples/usage_example.py")


if __name__ == "__main__":
    asyncio.run(main()) 