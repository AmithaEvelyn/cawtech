import google.generativeai as genai
from typing import List, Dict, Union, Optional
import numpy as np
import json
import os
import time
import logging
import faiss
from ..config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for FAISS indices
FAISSIndex = Union[faiss.IndexFlatL2, faiss.IndexIVFFlat, faiss.IndexHNSWFlat]

class EmbeddingService:
    def __init__(self):
        # Initialize Google Gemini
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set")
            
        try:
            logger.info("Initializing Gemini API...")
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini API: {str(e)}")
            raise
        
        # Initialize FAISS
        self.dimension = settings.EMBEDDING_DIMENSION
        self.index: FAISSIndex = self._create_faiss_index()
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        
        # Load existing data if available
        self.data_dir = settings.STORAGE_DIRECTORY
        self._load_data()
        logger.info("EmbeddingService initialized successfully")

    def _create_faiss_index(self) -> FAISSIndex:
        """Create a FAISS index based on configuration."""
        try:
            if settings.FAISS_INDEX_TYPE == "FlatL2":
                logger.info("Creating FlatL2 index")
                return faiss.IndexFlatL2(self.dimension)
            elif settings.FAISS_INDEX_TYPE == "IVF":
                logger.info("Creating IVF index")
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                return index
            elif settings.FAISS_INDEX_TYPE == "HNSW":
                logger.info("Creating HNSW index")
                return faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                logger.warning(f"Unknown index type {settings.FAISS_INDEX_TYPE}, using FlatL2")
                return faiss.IndexFlatL2(self.dimension)
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            return faiss.IndexFlatL2(self.dimension)

    def _load_data(self):
        """Load existing embeddings and metadata if available."""
        try:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir, exist_ok=True)
                logger.info(f"Created storage directory: {self.data_dir}")
                return

            # Load FAISS index
            index_file = os.path.join(self.data_dir, "faiss.index")
            if os.path.exists(index_file):
                logger.info("Loading existing FAISS index...")
                self.index = faiss.read_index(index_file)
                logger.info("FAISS index loaded successfully")
            
            # Load documents and metadata
            data_file = os.path.join(self.data_dir, "documents.json")
            if os.path.exists(data_file):
                logger.info("Loading existing documents and metadata...")
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])
                    self.metadata = data.get('metadata', [])
                logger.info(f"Loaded {len(self.documents)} documents from storage")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.index = self._create_faiss_index()
            self.documents = []
            self.metadata = []

    def _save_data(self):
        """Save embeddings and metadata."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Save FAISS index
            index_file = os.path.join(self.data_dir, "faiss.index")
            logger.info("Saving FAISS index...")
            faiss.write_index(self.index, index_file)
            
            # Save documents and metadata
            data_file = os.path.join(self.data_dir, "documents.json")
            logger.info("Saving documents and metadata...")
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.documents)} documents to storage")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

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
                
                if 'embedding' not in response:
                    logger.error(f"Invalid response from Gemini API: {response}")
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
            new_documents = []
            new_metadata = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {(len(chunks) + batch_size - 1)//batch_size}")
                
                batch_embeddings = self.generate_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Store documents and metadata for this batch
                for j, chunk in enumerate(batch):
                    new_documents.append(chunk)
                    new_metadata.append({
                        **metadata,
                        'chunk_index': i + j,
                        'document_id': document_id
                    })
            
            # Convert embeddings to numpy array and add to FAISS index
            new_embeddings = np.array(all_embeddings).astype('float32')
            
            # If using IVF index, train it if needed
            if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained and len(self.documents) > 100:
                logger.info("Training IVF index...")
                self.index.train(new_embeddings)
            
            # Add new embeddings to index
            if new_embeddings.size > 0:
                if isinstance(self.index, faiss.IndexIVFFlat):
                    ids = np.arange(self.index.ntotal, self.index.ntotal + len(new_embeddings))
                    self.index.add_with_ids(new_embeddings, ids)
                else:
                    self.index.add(new_embeddings)
            
            # Update documents and metadata
            self.documents.extend(new_documents)
            self.metadata.extend(new_metadata)
            
            # Save data
            self._save_data()
            logger.info(f"Successfully stored {len(chunks)} chunks for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error storing document chunks: {str(e)}")
            raise

    def search_similar_chunks(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar chunks using the query."""
        if self.index.ntotal == 0:
            logger.warning("No documents available for search")
            return []
            
        try:
            # Generate query embedding
            query_embeddings = self.generate_embeddings([query])
            if not query_embeddings or not query_embeddings[0]:
                logger.error("Failed to generate query embedding")
                return []
                
            query_embedding = np.array(query_embeddings[0]).astype('float32').reshape(1, -1)
            
            # Search in FAISS index
            k = min(n_results, self.index.ntotal)  # Ensure k is not larger than total vectors
            if k > 0:
                distances, indices = self.index.search(query_embedding, k)
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx != -1:  # FAISS returns -1 for empty slots
                        results.append({
                            'text': self.documents[idx],
                            'metadata': self.metadata[idx],
                            'distance': float(distances[0][i])  # FAISS returns L2 distances
                        })
                
                logger.info(f"Found {len(results)} similar chunks")
                return results
            else:
                logger.warning("No vectors in index to search")
                return []
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return [] 