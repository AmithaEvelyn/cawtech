import google.generativeai as genai
from typing import List, Dict, Optional, Any
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
            
        try:
            logger.info("Initializing Gemini API...")
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(model_name='gemini-pro')
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini API: {str(e)}")
            raise
            
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
                try:
                    conversation_history = self.memory_service.get_conversation_history(
                        conversation_id,
                        settings.SHORT_TERM_MEMORY_SIZE
                    )
                    logger.info(f"Retrieved conversation history with {len(conversation_history)} entries")
                except Exception as e:
                    logger.warning(f"Error retrieving conversation history: {str(e)}")
                    conversation_history = []

            # Search for relevant chunks
            try:
                relevant_chunks = self.embedding_service.search_similar_chunks(question)
                logger.info(f"Found {len(relevant_chunks)} relevant chunks")
                if relevant_chunks:
                    logger.info(f"First chunk preview: {relevant_chunks[0]['text'][:100]}...")
            except Exception as e:
                logger.error(f"Error searching for relevant chunks: {str(e)}")
                return {
                    'answer': "I encountered an error while searching for relevant information. Please try again.",
                    'confidence': 0.0,
                    'sources': [],
                    'conversation_id': conversation_id or str(uuid.uuid4())
                }
            
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
            logger.info(f"Prepared context with {len(context)} characters")
            
            # Prepare conversation history
            history_text = ""
            if conversation_history:
                history_text = "\n".join([
                    f"Q: {q}\nA: {a}" for q, a in conversation_history
                ])
                logger.info(f"Prepared conversation history with {len(history_text)} characters")

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
                logger.info("Generating answer using Gemini...")
                logger.info(f"Prompt length: {len(prompt)} characters")
                response = self.model.generate_content(prompt)
                
                if not response or not response.text:
                    logger.error("Empty response from Gemini")
                    return {
                        'answer': "I apologize, but I couldn't generate a proper response. Please try again.",
                        'confidence': 0.0,
                        'sources': [],
                        'conversation_id': conversation_id or str(uuid.uuid4())
                    }
                
                # Extract answer and confidence
                answer = response.text.strip()
                confidence = 0.8  # Default confidence for now
                
                # Store in memory
                if conversation_id:
                    try:
                        self.memory_service.add_to_conversation(
                            conversation_id,
                            question,
                            answer
                        )
                        logger.info("Stored conversation in memory")
                    except Exception as e:
                        logger.warning(f"Error storing conversation: {str(e)}")

                return {
                    'answer': answer,
                    'confidence': confidence,
                    'sources': [
                        {
                            'text': {
                                'content': chunk['text'],
                                'metadata': chunk['metadata']
                            }
                        }
                        for chunk in relevant_chunks
                    ],
                    'conversation_id': conversation_id or str(uuid.uuid4())
                }
            except Exception as e:
                logger.error(f"Error generating answer with Gemini: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error details: {str(e)}")
                return {
                    'answer': "I apologize, but I encountered an error while generating the answer. Please try again.",
                    'confidence': 0.0,
                    'sources': [],
                    'conversation_id': conversation_id or str(uuid.uuid4())
                }
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
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