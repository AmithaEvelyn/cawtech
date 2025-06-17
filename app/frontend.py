import streamlit as st
import requests
import json
import os
from typing import List, Dict
import uuid

# API endpoint
API_URL = "http://127.0.0.1:8000"

def main():
    st.title("Document Q&A System")
    
    # Initialize session state for conversation
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    # File upload
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt', 'html', 'md'])
    
    if uploaded_file is not None:
        # Save the file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Upload to API
            with open(temp_file_path, "rb") as f:
                response = requests.post(
                    f"{API_URL}/api/documents/upload",
                    files={"file": (uploaded_file.name, f)}
                )
            
            if response.status_code == 200:
                st.success("Document uploaded successfully!")
            else:
                st.error("Error uploading document")
        
        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    st.warning(f"Could not delete temporary file: {str(e)}")
    
    # Question answering
    st.header("Ask Questions")
    question = st.text_input("Enter your question:")
    
    if st.button("Ask"):
        if question:
            try:
                response = requests.post(
                    f"{API_URL}/api/qa/ask",
                    json={
                        "text": question,
                        "conversation_id": st.session_state.conversation_id
                    }
                )
                
                if response.status_code == 200:
                    answer_data = response.json()
                    
                    # Display answer
                    st.subheader("Answer")
                    st.write(answer_data.get('answer', 'No answer available'))
                    
                    # Display confidence
                    confidence = answer_data.get('confidence', 0.0)
                    st.write(f"Confidence: {confidence:.2%}")
                    
                    # Display sources
                    sources = answer_data.get('sources', [])
                    if sources:
                        st.subheader("Sources")
                        for source in sources:
                            try:
                                content = source.get('text', {}).get('content', '')
                                if content:
                                    st.write(f"- {content[:200]}...")
                            except Exception as e:
                                st.warning(f"Error displaying source: {str(e)}")
                    
                    # Update conversation history
                    st.session_state.conversation_id = answer_data.get('conversation_id', st.session_state.conversation_id)
                
                else:
                    error_msg = response.json().get('detail', 'Unknown error occurred')
                    st.error(f"Error getting answer: {error_msg}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Conversation history
    st.header("Conversation History")
    try:
        history_response = requests.get(
            f"{API_URL}/api/qa/history/{st.session_state.conversation_id}"
        )
        
        if history_response.status_code == 200:
            history = history_response.json()['history']
            for exchange in history:
                st.write(f"Q: {exchange['question']}")
                st.write(f"A: {exchange['answer']}")
                st.write("---")
    
    except Exception as e:
        st.error(f"Error loading conversation history: {str(e)}")

if __name__ == "__main__":
    main() 

