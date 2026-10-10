"""Legal Judgment Analysis API package."""
from api.index import (
    MAX_DOCUMENTS,
    MAX_UPLOAD_BYTES,
    QueryRequest,
    _documents,
    _documents_lock,
    app,
    delete_document,
    health,
    query_judgment,
    upload_document,
)

__all__ = [
    "app",
    "_documents",
    "_documents_lock",
    "QueryRequest",
    "health",
    "upload_document",
    "query_judgment",
    "delete_document",
    "MAX_UPLOAD_BYTES",
    "MAX_DOCUMENTS",
]
