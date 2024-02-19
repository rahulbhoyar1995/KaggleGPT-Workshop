import streamlit as st
from openai import OpenAI
from utils import download_the_conversation, summarise_the_conversation, generating_response
import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

st.set_page_config(
        page_title="KaggleGPT",
        page_icon='🤖',
        layout='centered',
        initial_sidebar_state='collapsed'
    )

st.header("KaggleGPT")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("New Conversation"):
        st.session_state.messages = []
    if st.button("Summarise the Conversation"):
        summarise_the_conversation()
    if st.button("Download the Conversation"):
        download_the_conversation()
   
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
#prompt = st.chat_input("What is up?")
embeddings = OpenAIEmbeddings()
loaded_db = FAISS.load_local("vector_database", embeddings)
retriever = loaded_db.as_retriever()

response = False
# user_input_query = st.text_input("Write the topic for which you want the Kaggle Datasets recommendations for?")

# # with st.chat_message("assistant"):
# string_response = generating_response(user_input_query,retriever)
# response = st.write(string_response)# Add assistant response to chat history
# st.session_state.messages.append({"role": "assistant", "content": string_response})




user_input_query = st.text_input("Write the topic for which you want Kaggle Datasets recommendations for?")

    # Button to trigger recommendation
if st.button("Get Recommendations"):
    string_response = generating_response(user_input_query,retriever)
    response = st.write(string_response)# Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": string_response})



response = True

prompt = st.chat_input("Ask me anything...")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            )
        response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        print("All messages :",st.session_state.messages)     