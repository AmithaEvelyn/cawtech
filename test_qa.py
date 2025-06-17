import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.document import DocumentProcessor
from app.services.embedding import EmbeddingService
from app.services.qa import QAService

client = TestClient(app)

def test_upload_document():
    """Test document upload and processing."""
    # Create a test document
    test_content = "This is a test document. It contains some sample text for testing purposes."
    test_file = "test_doc.txt"
    
    with open(test_file, "w") as f:
        f.write(test_content)
    
    try:
        # Upload the document
        with open(test_file, "rb") as f:
            response = client.post(
                "/api/documents/upload",
                files={"file": (test_file, f, "text/plain")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert "chunks" in data
        assert data["chunks"] > 0
    
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)

def test_ask_question():
    """Test question answering functionality."""
    # First upload a document
    test_content = "The capital of France is Paris. Paris is known for the Eiffel Tower."
    test_file = "test_doc.txt"
    
    with open(test_file, "w") as f:
        f.write(test_content)
    
    try:
        # Upload document
        with open(test_file, "rb") as f:
            upload_response = client.post(
                "/api/documents/upload",
                files={"file": (test_file, f, "text/plain")}
            )
        
        # Ask a question
        question = {
            "text": "What is the capital of France?",
            "conversation_id": None
        }
        
        response = client.post("/api/qa/ask", json=question)
        
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "confidence" in data
        assert "sources" in data
        assert "conversation_id" in data
        assert "Paris" in data["text"]
    
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)

def test_feedback():
    """Test feedback submission."""
    # Submit feedback
    feedback = {
        "answer_id": "test_answer_id",
        "rating": 5,
        "comment": "Great answer!"
    }
    
    response = client.post("/api/qa/feedback", json=feedback)
    assert response.status_code == 200
    
    # Get feedback stats
    stats_response = client.get("/api/qa/feedback/stats")
    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert "total_feedback" in stats
    assert "average_rating" in stats
    assert "rating_distribution" in stats

if __name__ == "__main__":
    pytest.main([__file__]) 