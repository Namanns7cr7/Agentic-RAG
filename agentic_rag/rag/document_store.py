import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4

from ..learning.document_processor import get_document_processor
from .utils_logger import get_logger

logger = get_logger(__name__)


class DocumentStore:
    """Manages document uploads, deduplication using SHA-256 file hashes, 

    and local metadata storage synchronization.
    """

    def __init__(self, upload_dir: str = "data/uploads", index_dir: str = "data/index"):
        self.upload_dir = Path(upload_dir)
        self.index_dir = Path(index_dir)
        self.metadata_path = self.index_dir / "metadata.json"

        # Create directories if they do not exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.metadata: Dict[str, Any] = {"hashes": {}, "documents": {}}
        self.load_metadata()

    def load_metadata(self) -> None:
        """Load document metadata from metadata.json."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    if "hashes" not in self.metadata:
                        self.metadata["hashes"] = {}
                    if "documents" not in self.metadata:
                        self.metadata["documents"] = {}
            except Exception as e:
                logger.error("Failed to load metadata.json: %s", e)

    def save_metadata(self) -> None:
        """Save document metadata to metadata.json."""
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save metadata.json: %s", e)

    def calculate_hash(self, file_bytes: bytes) -> str:
        """Calculate the SHA-256 hash of a file's content bytes."""
        return hashlib.sha256(file_bytes).hexdigest()

    def get_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get document details by file hash if it already exists."""
        return self.metadata["hashes"].get(file_hash)

    def add_document(
        self,
        filename: str,
        file_bytes: bytes,
        pipeline,  # Pipeline instance to add chunks to vector store
        chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """Process and add document to the store with SHA-256 deduplication."""
        file_hash = self.calculate_hash(file_bytes)

        # Check for deduplication
        existing = self.get_document_by_hash(file_hash)
        if existing:
            doc_id = existing["doc_id"]
            # Verify the document still exists in index
            if doc_id in self.metadata["documents"]:
                logger.info("Deduplicated file: %s (hash=%s)", filename, file_hash)
                return {
                    "doc_id": doc_id,
                    "filename": existing["filename"],
                    "total_characters": existing["total_characters"],
                    "total_chunks": existing["total_chunks"],
                    "status": "deduplicated",
                }

        # Process the new document
        doc_id = str(uuid4())
        processor = get_document_processor()
        
        # Save file to uploads folder
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (".", "_", "-"))
        save_path = self.upload_dir / f"{doc_id}_{safe_filename}"
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        # Chunk the document using the text processor
        try:
            result = processor.process(
                file_bytes,
                filename,
                chunk_strategy="sentence",
                chunk_size=chunk_size,
            )
        except Exception as e:
            # Clean up saved file if processing fails
            if save_path.exists():
                os.remove(save_path)
            raise e

        # Add to the RAG vector store
        chunks = result["chunks"]
        total_chunks = len(chunks)

        if chunks:
            # We construct rich metadata for each chunk
            metadatas = [
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "page_number": self._extract_page_number(chunk_text, i + 1),
                }
                for i, chunk_text in enumerate(chunks)
            ]
            
            # Embed and insert chunks to Pipeline's vector store
            embs = pipeline.embedder.encode(chunks)
            pipeline.store.add(chunks, embs, metadatas=metadatas)

            # Persist FAISS index changes
            faiss_path = self.index_dir / "faiss.index"
            pipeline.store.save(str(faiss_path))

        # Save metadata info
        doc_info = {
            "doc_id": doc_id,
            "filename": filename,
            "total_characters": result["total_chars"],
            "total_chunks": total_chunks,
            "file_path": str(save_path),
            "uploaded_at": Path(save_path).stat().st_mtime,
        }

        self.metadata["hashes"][file_hash] = doc_info
        self.metadata["documents"][doc_id] = doc_info
        self.save_metadata()

        return {
            "doc_id": doc_id,
            "filename": filename,
            "total_characters": result["total_chars"],
            "total_chunks": total_chunks,
            "status": "processed",
        }

    def delete_document(self, doc_id: str, pipeline) -> bool:
        """Delete a document from index, metadata, and local file storage."""
        if doc_id not in self.metadata["documents"]:
            return False

        doc_info = self.metadata["documents"][doc_id]
        
        # 1. Remove from local storage
        file_path = doc_info.get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error("Failed to delete local file %s: %s", file_path, e)

        # 2. Remove from vector index and save
        if hasattr(pipeline.store, "remove_doc"):
            pipeline.store.remove_doc(doc_id)
            faiss_path = self.index_dir / "faiss.index"
            pipeline.store.save(str(faiss_path))

        # 3. Clean from metadata JSON
        del self.metadata["documents"][doc_id]
        # Remove from hashes too
        hash_to_del = None
        for file_hash, info in self.metadata["hashes"].items():
            if info["doc_id"] == doc_id:
                hash_to_del = file_hash
                break
        if hash_to_del:
            del self.metadata["hashes"][hash_to_del]

        self.save_metadata()
        return True

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents currently indexed."""
        return list(self.metadata["documents"].values())

    def _extract_page_number(self, chunk_text: str, default_num: int) -> int:
        """Heuristic to extract page number if present in text (e.g., from PyMuPDF header)."""
        match = re.search(r"Page (\d+)", chunk_text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except:
                pass
        return default_num


import re
