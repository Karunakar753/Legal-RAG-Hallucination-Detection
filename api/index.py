"""Vercel Serverless Function entrypoint for Legal Judgment Analysis API."""
import importlib.util
import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load root api.py module
api_file = ROOT_DIR / "api.py"
spec = importlib.util.spec_from_file_location("legal_rag_api", str(api_file))
legal_rag_api = importlib.util.module_from_spec(spec)
sys.modules["legal_rag_api"] = legal_rag_api
spec.loader.exec_module(legal_rag_api)

app = legal_rag_api.app
_documents = legal_rag_api._documents
_documents_lock = legal_rag_api._documents_lock
QueryRequest = legal_rag_api.QueryRequest
health = legal_rag_api.health
upload_document = legal_rag_api.upload_document
query_judgment = legal_rag_api.query_judgment
delete_document = legal_rag_api.delete_document
MAX_UPLOAD_BYTES = legal_rag_api.MAX_UPLOAD_BYTES
MAX_DOCUMENTS = legal_rag_api.MAX_DOCUMENTS
