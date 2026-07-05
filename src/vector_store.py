import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class LocalVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the local vector store with a SentenceTransformer model.
        """
        # Load embedding model locally (will download on first run)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Generates embeddings for all chunks and builds a FAISS index.
        """
        if not chunks:
            return
            
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity (Inner Product on normalized vectors)
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        # IndexFlatIP uses Inner Product (which matches Cosine Similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        self.metadata = chunks

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """
        Queries the vector store.
        Returns a list of tuples: (chunk_metadata, cosine_similarity_score)
        """
        if self.index is None or not self.metadata:
            return []
            
        query_vector = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # FAISS returns -1 if index is not found or out of bounds
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
                
        return results

    def save(self, directory: str = ".vector_store") -> None:
        """
        Saves the FAISS index and metadata pickle file to a local directory.
        """
        if self.index is None:
            return
            
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "index.faiss")
        metadata_path = os.path.join(directory, "metadata.pkl")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata dict
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, directory: str = ".vector_store") -> bool:
        """
        Loads the FAISS index and metadata pickle file from a local directory.
        Returns True if successful, False otherwise.
        """
        index_path = os.path.join(directory, "index.faiss")
        metadata_path = os.path.join(directory, "metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return False
            
        try:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        except Exception:
            return False
