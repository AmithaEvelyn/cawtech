from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import logging
from typing import Optional
from .services.document import DocumentProcessor
from .services.embedding import EmbeddingService
from .services.qa import QAService
from .services.memory import MemoryService
from .models.schemas import Question, Answer, Feedback
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.PROJECT_NAME)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
qa_service = QAService()
memory_service = MemoryService()

# Create necessary directories
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/memory", exist_ok=True)
os.makedirs(settings.STORAGE_DIRECTORY, exist_ok=True)

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    # Save file temporarily
    file_path = os.path.join("data/uploads", file.filename)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document
        result = document_processor.process_document(file_path)
        
        if not result['chunks']:
            raise HTTPException(
                status_code=400,
                detail="No text content could be extracted from the document"
            )
        
        # Generate and store embeddings
        document_id = str(uuid.uuid4())
        embedding_service.store_document_chunks(
            document_id,
            result['chunks'],
            result['metadata']
        )
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "chunks": len(result['chunks'])
        }
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/qa/ask")
async def ask_question(question: Question):
    """Ask a question about the documents."""
    try:
        if not question.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
            
        answer = await qa_service.answer_question(
            question.text,
            question.conversation_id
        )
        
        # Validate the answer format
        try:
            return Answer(
                answer=answer['answer'],
                confidence=answer['confidence'],
                sources=answer['sources'],
                conversation_id=answer['conversation_id']
            )
        except Exception as e:
            logger.error(f"Error validating answer format: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error in answer format"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error answering question: {str(e)}"
        )

@app.get("/api/qa/history/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history."""
    try:
        history = memory_service.get_conversation_history(
            conversation_id,
            settings.CONVERSATION_HISTORY_SIZE
        )
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting conversation history: {str(e)}"
        )

@app.post("/api/qa/feedback")
async def submit_feedback(feedback: Feedback):
    """Submit feedback for an answer."""
    try:
        memory_service.add_feedback(
            feedback.answer_id,
            feedback.rating,
            feedback.comment
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting feedback: {str(e)}"
        )

@app.get("/api/qa/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics."""
    try:
        return memory_service.get_feedback_stats()
    except Exception as e:
        logger.error(f"Error getting feedback stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting feedback stats: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 