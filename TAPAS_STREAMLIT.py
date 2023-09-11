#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
from transformers import pipeline

# Install torch-scatter
#get_ipython().system('pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html')

# Install transformers
#get_ipython().system('pip install -q transformers==4.4.2')

# Load the TAPAS pipeline
tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")

def main():
    st.title("Table-based Question Answering Chatbot")
    
    # Initialize conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    conversation = st.session_state.conversation
    
    # Sidebar
    st.sidebar.header("Options")
    show_head = st.sidebar.checkbox("Show Table Head", value=True)
    
    # Drag and drop files
    uploaded_file = st.sidebar.file_uploader("Drag and drop a CSV file here", type=["csv"], key="fileUploader")
    
    if uploaded_file is not None:
        table = pd.read_csv(uploaded_file)
        table = table.astype(str)
        
        if show_head:
            st.write("Uploaded Table (Head):")
            st.dataframe(table.head())
        else:
            st.write("Uploaded Table:")
            st.dataframe(table)
        
        # Get user's question
        prompt = st.text_input("Ask a question about the table:", key="question_input")
        
        if prompt:
            # Perform question answering
            answer = tqa(table=table, query=prompt)["answer"]
            
            # Update conversation history
            conversation.append(("User:", prompt))
            conversation.append(("Chatbot:", answer))
            

        
        # Display conversation history
        st.write("Chat History:")
        for role, message in conversation:
            if role == "User:":
                st.markdown(f'<p><span style="color:red;">{role}</span> {message}</p>', unsafe_allow_html=True)
            elif role == "Chatbot:":
                st.markdown(f'<p><span style="color:green;">{role}</span> {message}</p>', unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()

