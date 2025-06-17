import google.generativeai as genai
from typing import List, Dict, Optional
import logging
from ..config import settings
from .embedding import EmbeddingService
from .memory import MemoryService
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAService:
    def __init__(self):
        # Initialize Google Gemini
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set")
            
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize services
        self.embedding_service = EmbeddingService()
        self.memory_service = MemoryService()
        
        logger.info("QAService initialized successfully")

    async def answer_question(
        self,
        question: str,
        conversation_id: Optional[str] = None
    ) -> Dict:
        """Answer a question using RAG and conversation history."""
        try:
            logger.info(f"Processing question: {question}")
            
            # Get conversation history if available
            conversation_history = []
            if conversation_id:
                conversation_history = self.memory_service.get_conversation_history(
                    conversation_id,
                    settings.SHORT_TERM_MEMORY_SIZE
                )
                logger.info(f"Retrieved conversation history with {len(conversation_history)} entries")

            # Search for relevant chunks
            relevant_chunks = self.embedding_service.search_similar_chunks(question)
            logger.info(f"Found {len(relevant_chunks)} relevant chunks")
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found for the question")
                return {
                    'answer': "I couldn't find any relevant information to answer your question. Please make sure you have uploaded some documents first.",
                    'confidence': 0.0,
                    'sources': [],
                    'conversation_id': conversation_id or str(uuid.uuid4())
                }
            
            # Prepare context from chunks
            context = "\n".join([chunk['text'] for chunk in relevant_chunks])
            
            # Prepare conversation history
            history_text = ""
            if conversation_history:
                history_text = "\n".join([
                    f"Q: {q}\nA: {a}" for q, a in conversation_history
                ])

            # Construct prompt with better instructions
            prompt = f"""You are an AI assistant that answers questions based on the provided context. 
If the answer cannot be found in the context, say so clearly.

Context information:
---------------------
{context}
---------------------

Previous conversation:
{history_text}

Current question: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the answer is not in the context, say "I cannot find the answer in the provided documents"
3. Be concise but informative
4. If you need to make assumptions, state them clearly
5. If the question is unclear, ask for clarification

Please provide your answer:"""

            # Generate answer
            try:
                logger.info("Generating answer using Gemini")
                response = self.model.generate_content(
                    contents=[prompt],
                    generation_config={
                        'temperature': 0.7,
                        'top_p': 0.8,
                        'top_k': 40,
                        'max_output_tokens': 1024,
                    }
                )
                answer = response.text
                logger.info("Successfully generated answer")
            except Exception as e:
                logger.error(f"Error generating answer: {str(e)}")
                answer = "I apologize, but I encountered an error while generating the answer. Please try again."

            # Store in memory
            if conversation_id:
                self.memory_service.add_to_conversation(
                    conversation_id,
                    question,
                    answer
                )
                logger.info("Stored conversation in memory")

            return {
                'answer': answer,
                'confidence': self._calculate_confidence(relevant_chunks),
                'sources': [
                    {
                        'text': chunk['text'],
                        'metadata': chunk['metadata']
                    }
                    for chunk in relevant_chunks
                ],
                'conversation_id': conversation_id or str(uuid.uuid4())
            }
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            return {
                'answer': "I apologize, but I encountered an error while processing your question. Please try again.",
                'confidence': 0.0,
                'sources': [],
                'conversation_id': conversation_id or str(uuid.uuid4())
            }

    def _calculate_confidence(self, chunks: List[Dict]) -> float:
        """Calculate confidence score based on chunk relevance."""
        if not chunks:
            return 0.0
        
        try:
            # Calculate average distance (lower is better)
            avg_distance = sum(chunk['distance'] for chunk in chunks) / len(chunks)
            
            # Convert to confidence score (0-1)
            confidence = 1.0 - min(avg_distance, 1.0)
            return confidence
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0 