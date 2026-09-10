import os
import uuid
import logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger("MemoryManager")

class MemoryManager:
    def __init__(self, db_path: str = None):
        if not db_path:
            db_path = str(Path.home() / ".cache" / "mellm_db")
        
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        
        logger.info(f"Initializing MemoryManager at {self.db_path}")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Default embedding function uses all-MiniLM-L6-v2 which is fast on CPU
        # Uses chromadb's ONNX runtime to bypass sentence-transformers/scipy DLL restrictions
        try:
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            logger.error(f"Failed to load default embedding function: {e}")
            self.embedding_fn = None
            
        self.memory_collection = self.client.get_or_create_collection(
            name="memory_logs",
            embedding_function=self.embedding_fn
        )
        
        # Use an ephemeral client for documents so they don't persist across restarts
        self.ephemeral_client = chromadb.EphemeralClient()
        self.document_collection = self.ephemeral_client.get_or_create_collection(
            name="session_documents",
            embedding_function=self.embedding_fn
        )
        logger.info("ChromaDB collections loaded successfully.")

    def save_memory(self, text: str, source: str = "conversation"):
        """Saves a fact or log into the long-term memory."""
        if not text.strip():
            return
            
        doc_id = f"mem_{uuid.uuid4().hex[:8]}"
        self.memory_collection.add(
            documents=[text],
            metadatas=[{"source": source}],
            ids=[doc_id]
        )
        logger.info(f"Saved memory: {text[:50]}...")

    def retrieve_memory(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieves relevant facts from long-term memory."""
        if self.memory_collection.count() == 0:
            return []
            
        k = min(top_k, self.memory_collection.count())
        results = self.memory_collection.query(
            query_texts=[query],
            n_results=k
        )
        
        if results["documents"] and results["documents"][0]:
            return results["documents"][0]
        return []

    def ingest_document(self, text: str, filename: str):
        """Chunks a document and saves it into the documents collection."""
        # Chunk by characters to ensure hard limits even if PDF has no spaces
        chunk_size = 1500
        overlap = 200
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
                
        if not chunks:
            return
            
        ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in range(len(chunks))]
        metadatas = [{"filename": filename, "chunk": i} for i in range(len(chunks))]
        
        self.document_collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Ingested document {filename} into {len(chunks)} chunks.")

    def retrieve_document_context(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieves relevant document chunks."""
        # if collection is empty, chromadb throws error on query if n_results > count
        if self.document_collection.count() == 0:
            return []
            
        k = min(top_k, self.document_collection.count())
        results = self.document_collection.query(
            query_texts=[query],
            n_results=k
        )
        
        if results["documents"] and results["documents"][0]:
            return results["documents"][0]
        return []

    def clear_documents(self):
        """Clears the ephemeral document collection for the current session."""
        try:
            self.ephemeral_client.delete_collection("session_documents")
            self.document_collection = self.ephemeral_client.get_or_create_collection(
                name="session_documents",
                embedding_function=self.embedding_fn
            )
            logger.info("Cleared session documents.")
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
