import os
from typing import List, Dict
import PyPDF2
from docx import Document as DocxDocument
import markdown
from bs4 import BeautifulSoup
from ..config import settings

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = {
            'pdf': self._process_pdf,
            'docx': self._process_docx,
            'txt': self._process_txt,
            'html': self._process_html,
            'md': self._process_markdown
        }

    def process_document(self, file_path: str) -> Dict:
        """Process a document and return its chunks and metadata."""
        file_ext = os.path.splitext(file_path)[1].lower()[1:]
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        content = self.supported_formats[file_ext](file_path)
        chunks = self._chunk_text(content)
        
        return {
            'chunks': chunks,
            'metadata': {
                'filename': os.path.basename(file_path),
                'content_type': file_ext,
                'total_chunks': len(chunks)
            }
        }

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + settings.CHUNK_SIZE, text_length)
            chunk = text[start:end]
            
            # Adjust chunk boundaries to not break words
            if end < text_length:
                last_space = chunk.rfind(' ')
                if last_space != -1:
                    end = start + last_space
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - settings.CHUNK_OVERLAP
            
            if len(chunks) >= settings.MAX_CHUNKS_PER_DOCUMENT:
                break
        
        return chunks

    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _process_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        doc = DocxDocument(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def _process_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def _process_html(self, file_path: str) -> str:
        """Extract text from HTML file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            return soup.get_text()

    def _process_markdown(self, file_path: str) -> str:
        """Extract text from Markdown file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            html = markdown.markdown(file.read())
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text() 