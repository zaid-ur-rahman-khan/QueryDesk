"""
QueryDesk MCP server.
Format-aware document tools plus shared semantic RAG retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from docx import Document as DocxDocument
from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader

from create_vector import (
    UPLOAD_DIR,
    SUPPORTED_EXTENSIONS,
    list_documents as indexed_documents,
    load_file,
    search_documents as vector_search,
)

mcp = FastMCP("QueryDesk Document Server")


def _resolve_document(filename: str) -> Path:
    # Restrict file access to the configured upload directory.
    safe_name = Path(filename).name
    if safe_name != filename:
        raise ValueError("Use a filename only; directory paths are not accepted.")
    path = UPLOAD_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {safe_name}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {path.suffix}")
    return path


def _format_hits(documents) -> str:
    if not documents:
        return "No matching document content was found."
    output = []
    for doc in documents:
        meta = doc.metadata
        source = meta.get("filename", meta.get("source", "unknown"))
        location = []
        if meta.get("page_number") is not None:
            location.append(f"page {meta['page_number']}")
        if meta.get("sheet_name"):
            location.append(f"sheet {meta['sheet_name']}")
        if meta.get("row_start") is not None:
            location.append(f"rows {meta.get('row_start')}-{meta.get('row_end')}")
        suffix = f" ({', '.join(location)})" if location else ""
        output.append(f"[{source}{suffix}]\n{doc.page_content}")
    return "\n\n---\n\n".join(output)


@mcp.tool()
def search_documents(question: str, filename: str | None = None, top_k: int = 5) -> str:
    """Semantic RAG search across the indexed library, optionally restricted to one filename."""
    k = max(1, min(int(top_k), 12))
    if filename:
        _resolve_document(filename)
        hits = vector_search(question, k=k, filter={"filename": Path(filename).name})
    else:
        hits = vector_search(question, k=k)
    return _format_hits(hits)


@mcp.tool()
def get_pdf_pages(filename: str, start_page: int, end_page: int | None = None) -> str:
    """Retrieve exact page text from a PDF. Page numbers are one-based and inclusive."""
    path = _resolve_document(filename)
    if path.suffix.lower() != ".pdf":
        raise ValueError("get_pdf_pages only accepts PDF files.")
    reader = PdfReader(str(path))
    last_page = end_page if end_page is not None else start_page
    if start_page < 1 or last_page < start_page or last_page > len(reader.pages):
        raise ValueError(f"Valid page range is 1 to {len(reader.pages)}.")
    pages = []
    for page_number in range(start_page, last_page + 1):
        text = reader.pages[page_number - 1].extract_text() or ""
        pages.append(f"[{filename} — page {page_number}]\n{text}")
    return "\n\n".join(pages)


@mcp.tool()
def search_text_document(filename: str, query: str, max_matches: int = 8) -> str:
    """Search within a DOCX or TXT file and return matching paragraphs/lines."""
    path = _resolve_document(filename)
    extension = path.suffix.lower()
    if extension == ".docx":
        paragraphs = [p.text for p in DocxDocument(str(path)).paragraphs if p.text.strip()]
    elif extension == ".txt":
        paragraphs = path.read_text(encoding="utf-8", errors="replace").splitlines()
    else:
        raise ValueError("This tool accepts DOCX and TXT documents only.")

    terms = [term.lower() for term in query.split() if term.strip()]
    matches = [
        f"Line/paragraph {index + 1}: {text}"
        for index, text in enumerate(paragraphs)
        if all(term in text.lower() for term in terms)
    ][:max(1, min(int(max_matches), 30))]
    return "\n".join(matches) if matches else "No matching paragraphs or lines found."


@mcp.tool()
def read_spreadsheet(filename: str, sheet_name: str | None = None, max_rows: int = 50) -> str:
    """Read CSV or Excel data, optionally selecting an Excel sheet; returns a bounded preview."""
    path = _resolve_document(filename)
    if path.suffix.lower() == ".csv":
        if sheet_name:
            raise ValueError("CSV files do not have sheet names.")
        frame = pd.read_csv(path)
        selected_name = "CSV"
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        if sheet_name is None:
            selected_name = next(iter(sheets))
        elif sheet_name in sheets:
            selected_name = sheet_name
        else:
            raise ValueError(f"Sheet not found. Available sheets: {list(sheets)}")
        frame = sheets[selected_name]
    else:
        raise ValueError("This tool accepts CSV and Excel files only.")

    preview = frame.head(max(1, min(int(max_rows), 100)))
    return json.dumps(
        {
            "filename": filename,
            "sheet": selected_name,
            "columns": [str(column) for column in frame.columns],
            "total_rows": int(len(frame)),
            "preview": preview.fillna("").to_dict(orient="records"),
        },
        default=str,
        ensure_ascii=False,
    )


@mcp.tool()
def analyze_spreadsheet(
    filename: str,
    operation: Literal["describe", "sum", "mean", "count", "min", "max", "filter", "groupby"],
    column: str | None = None,
    filter_column: str | None = None,
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains"] | None = None,
    value: str | float | None = None,
    group_by: str | None = None,
    aggregation: Literal["sum", "mean", "count", "min", "max"] = "sum",
    sheet_name: str | None = None,
) -> str:
    """Perform real Pandas calculations or controlled filtering/grouping on CSV/Excel data."""
    path = _resolve_document(filename)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        if sheet_name is None:
            frame = next(iter(sheets.values()))
        elif sheet_name in sheets:
            frame = sheets[sheet_name]
        else:
            raise ValueError(f"Sheet not found. Available sheets: {list(sheets)}")
    else:
        raise ValueError("This tool accepts CSV and Excel files only.")

    if operation == "describe":
        result = frame.describe(include="all").fillna("")
    elif operation == "filter":
        if not filter_column or filter_column not in frame.columns:
            raise ValueError(f"Choose a valid filter_column from {list(frame.columns)}")
        if operator is None or value is None:
            raise ValueError("Filter requires operator and value.")
        series = frame[filter_column]
        if operator == "contains":
            mask = series.astype(str).str.contains(str(value), case=False, na=False)
        else:
            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            numeric_series = pd.to_numeric(series, errors="coerce")
            if pd.notna(numeric_value) and numeric_series.notna().any():
                series, value = numeric_series, numeric_value
            comparisons = {
                "eq": lambda: series == value,
                "ne": lambda: series != value,
                "gt": lambda: series > value,
                "gte": lambda: series >= value,
                "lt": lambda: series < value,
                "lte": lambda: series <= value,
            }
            mask = comparisons[operator]()
        result = frame.loc[mask].head(200)
    elif operation == "groupby":
        if not group_by or group_by not in frame.columns:
            raise ValueError(f"Choose a valid group_by column from {list(frame.columns)}")
        if aggregation != "count" and (not column or column not in frame.columns):
            raise ValueError(f"Choose a valid numeric aggregation column from {list(frame.columns)}")
        grouped = frame.groupby(group_by, dropna=False)
        if aggregation == "count":
            result = grouped.size().reset_index(name="count")
        else:
            numeric = frame.copy()
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
            result = getattr(numeric.groupby(group_by, dropna=False)[column], aggregation)().reset_index()
    elif operation == "count" and column is None:
        result = pd.DataFrame({"count": [int(len(frame))]})
    else:
        if not column or column not in frame.columns:
            raise ValueError(f"Choose a valid column from {list(frame.columns)}")
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if operation == "count":
            result = pd.DataFrame({"count": [int(frame[column].count())]})
        elif operation == "sum":
            result = pd.DataFrame({"sum": [numeric.sum()]})
        elif operation == "mean":
            result = pd.DataFrame({"mean": [numeric.mean()]})
        elif operation == "min":
            result = pd.DataFrame({"min": [numeric.min()]})
        elif operation == "max":
            result = pd.DataFrame({"max": [numeric.max()]})
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    return json.dumps(
        {"filename": filename, "operation": operation, "result": result.to_dict(orient="records")},
        default=str,
        ensure_ascii=False,
    )


@mcp.tool()
def list_documents() -> str:
    """List all documents currently indexed in the persistent QueryDesk library."""
    return json.dumps(indexed_documents(), ensure_ascii=False)


@mcp.tool()
def compare_documents(filenames: list[str], question: str) -> str:
    """Retrieve relevant evidence from two or more named documents for a grounded comparison."""
    if len(filenames) < 2:
        raise ValueError("Provide at least two filenames to compare.")
    results = []
    for filename in filenames[:5]:
        _resolve_document(filename)
        hits = vector_search(
            question, k=5, filter={"filename": Path(filename).name}
        )
        results.append(f"DOCUMENT: {filename}\n{_format_hits(hits)}")
    return "\n\n====================\n\n".join(results)


@mcp.tool()
def get_document_summary(filename: str, section_query: str | None = None) -> str:
    """Retrieve content for a whole-document or section summary; Gemini writes the summary."""
    _resolve_document(filename)
    query = section_query or f"Main topics, purpose, key facts and conclusions of {filename}"
    hits = vector_search(
        query, k=8, filter={"filename": Path(filename).name}
    )
    return _format_hits(hits)


if __name__ == "__main__":
    mcp.run()
