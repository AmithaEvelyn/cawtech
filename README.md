# Intelligent Document Q&A System

A production-ready document question-answering system that learns from user interactions and improves over time.

## Features

- Multi-format document support (PDF, DOCX, TXT, HTML, Markdown)
- Intelligent document chunking and embedding
- Context-aware query processing
- Conversation history management
- Feedback-based learning system

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

3. Run the application:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- POST `/api/documents/upload` - Upload and process documents
- POST `/api/qa/ask` - Ask questions about documents
- GET `/api/qa/history` - Get conversation history
- POST `/api/qa/feedback` - Provide feedback on answers

## Project Structure

```
app/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration settings
├── models/             # Data models
├── services/           # Business logic
│   ├── document.py     # Document processing
│   ├── embedding.py    # Embedding generation
│   ├── qa.py          # Q&A engine
│   └── memory.py      # Memory management
└── utils/             # Utility functions
``` 