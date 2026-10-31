"""
QueryDesk document ingestion and FAISS retrieval.

Responsibilities:
- Load PDF, DOCX, TXT, CSV, XLSX and XLS files from a persistent directory.
- Preserve source metadata (PDF page, spreadsheet sheet and row range).
- Split documents into chunks and embed them with Gemini.
- Persist a FAISS index and a small ingestion manifest.
- Incrementally add changed/new files and remove deleted files.

This module intentionally performs no ingestion at import time.
Run it directly to index the contents of UPLOAD_DIR.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("QUERYDESK_UPLOAD_DIR", BASE_DIR / "assets" / "uploaded_documents"))
VECTORSTORE_DIR = Path(os.getenv("QUERYDESK_VECTORSTORE_DIR", BASE_DIR / "vectorstore"))
MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls"}

CHUNK_SIZE = int(os.getenv("QUERYDESK_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("QUERYDESK_CHUNK_OVERLAP", "120"))
DEFAULT_TOP_K = int(os.getenv("QUERYDESK_TOP_K", "4"))
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")


# ---------------------------------------------------------------------------
# Gemini embeddings
# ---------------------------------------------------------------------------

class GeminiRetrievalEmbeddings(Embeddings):
    """
    LangChain adapter that uses the correct Gemini task type for each side
    of asymmetric retrieval: RETRIEVAL_DOCUMENT for indexed chunks and
    RETRIEVAL_QUERY for user questions.
    """

    def __init__(self, google_api_key: str | None = None, model: str = EMBEDDING_MODEL):
        api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Google API key not found. Set GOOGLE_API_KEY in your .env file."
            )

        self.client = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed_documents(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
        )

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(
            text,
            task_type="RETRIEVAL_QUERY",
        )


# Kept as a module-level object for easy reuse by server.py in Phase 2.
# Construction does not make an API request; it only validates configuration.
def get_embeddings() -> GeminiRetrievalEmbeddings:
    return GeminiRetrievalEmbeddings()


# Backward-compatible name for the original server.py import:
# `from create_vector import embeddings`. It remains None when no Gemini key
# is configured; actual ingestion/retrieval then raises a clear configuration
# error through get_embeddings().
_google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
embeddings = GeminiRetrievalEmbeddings(_google_key) if _google_key else None


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _base_metadata(file_path: Path) -> dict:
    return {
        "source": file_path.name,
        "filename": file_path.name,
        "file_type": file_path.suffix.lower().lstrip("."),
        "full_path": str(file_path.resolve()),
    }


def _load_pdf(file_path: Path) -> list[Document]:
    documents = PyPDFLoader(str(file_path)).load()
    for document in documents:
        # PyPDFLoader uses zero-based page metadata. Keep its original value
        # and add a human-friendly one-based page number for citations.
        page_index = document.metadata.get("page")
        document.metadata.update(_base_metadata(file_path))
        if page_index is not None:
            document.metadata["page_index"] = int(page_index)
            document.metadata["page_number"] = int(page_index) + 1
    return documents


def _load_docx(file_path: Path) -> list[Document]:
    documents = Docx2txtLoader(str(file_path)).load()
    for document in documents:
        document.metadata.update(_base_metadata(file_path))
    return documents


def _load_txt(file_path: Path) -> list[Document]:
    # Try common encodings so ordinary Windows-created text files are usable.
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            documents = TextLoader(str(file_path), encoding=encoding).load()
            for document in documents:
                document.metadata.update(_base_metadata(file_path))
            return documents
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode {file_path.name}: {last_error}")


def _read_csv(file_path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode CSV {file_path.name}: {last_error}")


def _dataframe_to_documents(
    dataframe: pd.DataFrame,
    file_path: Path,
    sheet_name: str,
    rows_per_document: int = 20,
) -> list[Document]:
    """Represent tabular data as searchable row groups with source metadata."""
    if dataframe.empty:
        return []

    # Normalize missing values and keep column names visible in each row.
    dataframe = dataframe.fillna("")
    documents: list[Document] = []

    for start in range(0, len(dataframe), rows_per_document):
        end = min(start + rows_per_document, len(dataframe))
        lines = []
        for row_number, (_, row) in enumerate(
            dataframe.iloc[start:end].iterrows(),
            start=start + 1,
        ):
            values = [
                f"{column}: {value}"
                for column, value in row.items()
                if str(value).strip()
            ]
            lines.append(f"Row {row_number}: " + " | ".join(values))

        metadata = _base_metadata(file_path)
        metadata.update(
            {
                "sheet_name": sheet_name,
                "row_start": start + 1,
                "row_end": end,
            }
        )
        documents.append(
            Document(
                page_content="\n".join(lines),
                metadata=metadata,
            )
        )

    return documents


def _load_csv(file_path: Path) -> list[Document]:
    dataframe = _read_csv(file_path)
    return _dataframe_to_documents(dataframe, file_path, sheet_name="CSV")


def _load_excel(file_path: Path) -> list[Document]:
    # sheet_name=None returns a mapping of sheet names to DataFrames.
    sheets = pd.read_excel(file_path, sheet_name=None)
    documents: list[Document] = []
    for sheet_name, dataframe in sheets.items():
        documents.extend(
            _dataframe_to_documents(dataframe, file_path, sheet_name=str(sheet_name))
        )
    return documents


def load_file(file_path: str | Path) -> list[Document]:
    """Load one supported file and return LangChain Documents with metadata."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document does not exist: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    loaders = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".txt": _load_txt,
        ".csv": _load_csv,
        ".xlsx": _load_excel,
        ".xls": _load_excel,
    }
    documents = loaders[extension](path)

    if not documents:
        raise ValueError(f"No readable text or rows found in {path.name}")

    return documents


# ---------------------------------------------------------------------------
# Chunking and persistent index helpers
# ---------------------------------------------------------------------------

def split_documents(documents: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(list(documents))


def _file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"documents": {}}
    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_manifest(manifest: dict) -> None:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = MANIFEST_PATH.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    temporary_path.replace(MANIFEST_PATH)


def _index_exists() -> bool:
    return (VECTORSTORE_DIR / "index.faiss").exists() and (
        VECTORSTORE_DIR / "index.pkl"
    ).exists()


def load_vector_store() -> FAISS:
    """Load the locally generated FAISS index."""
    if not _index_exists():
        raise FileNotFoundError(
            "FAISS index not found. Upload documents and run ingest_documents() first."
        )

    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings or get_embeddings(),
        # This index is generated and stored locally by QueryDesk. Never load
        # an index copied from an untrusted source with this setting enabled.
        allow_dangerous_deserialization=True,
    )


def _chunk_ids(filename: str, file_hash: str, chunk_count: int) -> list[str]:
    # Stable for a particular file version; changed files receive new IDs.
    return [
        hashlib.sha256(
            f"{filename}:{file_hash}:{chunk_number}".encode("utf-8")
        ).hexdigest()
        for chunk_number in range(chunk_count)
    ]


def ingest_documents(
    file_paths: Iterable[str | Path] | None = None,
) -> dict:
    """
    Incrementally synchronize the persistent FAISS index with UPLOAD_DIR.

    - New files are embedded and added.
    - Changed files have their previous chunks deleted and new chunks added.
    - Files removed from the upload directory are removed from the index.
    - Unchanged files are skipped.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    if file_paths is None:
        paths = sorted(
            path for path in UPLOAD_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        sync_removed_files = True
    else:
        paths = [Path(path) for path in file_paths]
        sync_removed_files = False

    manifest = _load_manifest()
    indexed_documents = manifest.setdefault("documents", {})

    vector_store = load_vector_store() if _index_exists() else None
    if vector_store is None and indexed_documents:
        # Manifest without the matching FAISS files is not a usable index.
        # Re-index current files instead of treating them as already embedded.
        indexed_documents.clear()
    added, updated, skipped, removed = [], [], [], []

    current_filenames = {path.name for path in paths}

    # Remove entries for files that have disappeared from the persistent
    # upload directory. This is only done during a full directory sync.
    if sync_removed_files:
        missing_names = set(indexed_documents) - current_filenames
        for filename in sorted(missing_names):
            old_ids = indexed_documents[filename].get("ids", [])
            if vector_store is not None and old_ids:
                vector_store.delete(ids=old_ids)
            indexed_documents.pop(filename, None)
            removed.append(filename)

    embeddings = None

    for path in paths:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Document does not exist: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        file_hash = _file_sha256(path)
        existing = indexed_documents.get(path.name)

        if existing and existing.get("sha256") == file_hash:
            skipped.append(path.name)
            continue

        # Parse and chunk before deleting the previous version. If parsing
        # fails, the old indexed version remains intact.
        loaded_documents = load_file(path)
        chunks = split_documents(loaded_documents)
        if not chunks:
            raise ValueError(f"No indexable chunks found in {path.name}")

        new_ids = _chunk_ids(path.name, file_hash, len(chunks))

        if vector_store is not None and existing:
            old_ids = existing.get("ids", [])
            if old_ids:
                vector_store.delete(ids=old_ids)

        if vector_store is None:
            embeddings = get_embeddings()
            vector_store = FAISS.from_documents(
                chunks,
                embeddings,
                ids=new_ids,
            )
        else:
            vector_store.add_documents(chunks, ids=new_ids)

        indexed_documents[path.name] = {
            "sha256": file_hash,
            "file_type": path.suffix.lower().lstrip("."),
            "chunk_count": len(chunks),
            "ids": new_ids,
        }

        if existing:
            updated.append(path.name)
        else:
            added.append(path.name)

    if vector_store is not None:
        vector_store.save_local(str(VECTORSTORE_DIR))

    _save_manifest(manifest)

    return {
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "removed": removed,
        "total_documents": len(indexed_documents),
        "total_chunks": sum(
            item.get("chunk_count", 0) for item in indexed_documents.values()
        ),
        "index_path": str(VECTORSTORE_DIR),
    }


def remove_document(filename: str, delete_file: bool = True) -> bool:
    """
    Remove one document from FAISS and the manifest.
    By default the physical upload is deleted too, preventing a later full
    directory sync from indexing the same document again. Pass False to keep it.
    """
    manifest = _load_manifest()
    entry = manifest.get("documents", {}).get(filename)
    if entry is None:
        return False

    if _index_exists():
        vector_store = load_vector_store()
        ids = entry.get("ids", [])
        if ids:
            vector_store.delete(ids=ids)

        # FAISS can persist an empty index, but removing the empty index makes
        # the next first ingestion path deterministic.
        if vector_store.index.ntotal == 0:
            for index_file in (
                VECTORSTORE_DIR / "index.faiss",
                VECTORSTORE_DIR / "index.pkl",
            ):
                if index_file.exists():
                    index_file.unlink()
        else:
            vector_store.save_local(str(VECTORSTORE_DIR))

    manifest["documents"].pop(filename, None)
    _save_manifest(manifest)

    if delete_file:
        uploaded_path = UPLOAD_DIR / filename
        if uploaded_path.exists() and uploaded_path.is_file():
            uploaded_path.unlink()

    return True


def list_documents() -> list[dict]:
    """Return indexed document metadata without loading the FAISS vectors."""
    manifest = _load_manifest()
    results = []
    for filename, details in manifest.get("documents", {}).items():
        results.append(
            {
                "filename": filename,
                "file_type": details.get("file_type"),
                "chunk_count": details.get("chunk_count", 0),
            }
        )
    return sorted(results, key=lambda item: item["filename"].lower())


def search_documents(
    query: str,
    k: int = DEFAULT_TOP_K,
    filter: dict | None = None,
) -> list[Document]:
    """Perform semantic similarity retrieval, optionally restricted by metadata."""
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")
    vector_store = load_vector_store()
    return vector_store.similarity_search(
        query,
        k=max(1, int(k)),
        filter=filter,
    )


# ---------------------------------------------------------------------------
# Standalone ingestion entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = ingest_documents()
    print("QueryDesk ingestion complete.")
    print(json.dumps(result, indent=2, ensure_ascii=False))
