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
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Upload to API
        try:
            with open(uploaded_file.name, "rb") as f:
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
            if os.path.exists(uploaded_file.name):
                os.remove(uploaded_file.name)
    
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
                    st.write(answer_data['text'])
                    
                    # Display confidence
                    st.write(f"Confidence: {answer_data['confidence']:.2%}")
                    
                    # Display sources
                    if answer_data['sources']:
                        st.subheader("Sources")
                        for source in answer_data['sources']:
                            st.write(f"- {source['text'][:200]}...")
                    
                    # Feedback
                    st.subheader("Feedback")
                    rating = st.slider("Rate this answer", 1, 5, 3)
                    comment = st.text_area("Comments (optional)")
                    
                    if st.button("Submit Feedback"):
                        feedback_response = requests.post(
                            f"{API_URL}/api/qa/feedback",
                            json={
                                "answer_id": answer_data.get('answer_id', ''),
                                "rating": rating,
                                "comment": comment
                            }
                        )
                        if feedback_response.status_code == 200:
                            st.success("Thank you for your feedback!")
                
                else:
                    st.error("Error getting answer")
            
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