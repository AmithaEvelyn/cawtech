import google.generativeai as genai
from typing import List, Dict
import numpy as np
import json
import os
import time
import logging
from ..config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        # Initialize Google Gemini
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set")
            
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize storage
        self.dimension = 768  # Gemini embedding dimension
        self.embeddings = []
        self.documents = []
        self.metadata = []
        
        # Load existing data if available
        self.data_file = os.path.join(settings.CHROMA_PERSIST_DIRECTORY, "vector_store.json")
        self._load_data()
        logger.info("EmbeddingService initialized successfully")

    def _load_data(self):
        """Load existing embeddings and metadata if available."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.embeddings = np.array(data['embeddings'])
                    self.documents = data['documents']
                    self.metadata = data['metadata']
                logger.info(f"Loaded {len(self.documents)} documents from storage")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.embeddings = []
            self.documents = []
            self.metadata = []

    def _save_data(self):
        """Save embeddings and metadata."""
        try:
            os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            with open(self.data_file, 'w') as f:
                json.dump({
                    'embeddings': self.embeddings.tolist(),
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)
            logger.info(f"Saved {len(self.documents)} documents to storage")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Gemini."""
        embeddings = []
        for text in texts:
            try:
                # Add timeout to prevent hanging
                start_time = time.time()
                
                # Use the correct Gemini embedding API
                response = genai.embed_content(
                    model='models/embedding-001',
                    content=text,
                    task_type='retrieval_document'
                )
                
                # Check if we've been processing too long
                if time.time() - start_time > 30:  # 30 second timeout
                    logger.warning("Embedding generation timed out")
                    embeddings.append([0.0] * self.dimension)
                    continue
                    
                embeddings.append(response['embedding'])
                logger.debug(f"Generated embedding for text of length {len(text)}")
            except Exception as e:
                logger.error(f"Embedding error: {str(e)}")
                # Fallback to zero vector if embedding fails
                embeddings.append([0.0] * self.dimension)
        return embeddings

    def store_document_chunks(self, document_id: str, chunks: List[str], metadata: Dict):
        """Store document chunks with their embeddings."""
        if not chunks:
            logger.warning("No chunks to embed")
            return
            
        try:
            # Process chunks in smaller batches to prevent timeouts
            batch_size = 5
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {(len(chunks) + batch_size - 1)//batch_size}")
                
                batch_embeddings = self.generate_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Store documents and metadata for this batch
                for j, chunk in enumerate(batch):
                    self.documents.append(chunk)
                    self.metadata.append({
                        **metadata,
                        'chunk_index': i + j,
                        'document_id': document_id
                    })
            
            # Convert embeddings to numpy array
            new_embeddings = np.array(all_embeddings)
            
            # Add to storage
            if len(self.embeddings) == 0:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            # Save data
            self._save_data()
            logger.info(f"Successfully stored {len(chunks)} chunks for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error storing document chunks: {str(e)}")
            raise

    def search_similar_chunks(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar chunks using the query."""
        if len(self.embeddings) == 0:
            logger.warning("No documents available for search")
            return []
            
        try:
            query_embedding = self.generate_embeddings([query])[0]
            query_array = np.array([query_embedding])
            
            # Calculate cosine similarity
            normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            normalized_query = query_array / np.linalg.norm(query_array)
            similarities = np.dot(normalized_embeddings, normalized_query.T).flatten()
            
            # Get top k results
            top_k_indices = np.argsort(similarities)[-n_results:][::-1]
            
            results = [
                {
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(1 - similarities[idx])  # Convert similarity to distance
                }
                for idx in top_k_indices
            ]
            
            logger.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return [] 