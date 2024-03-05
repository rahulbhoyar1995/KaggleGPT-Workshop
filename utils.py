import os
import streamlit as st
from utils_archieve import download_the_conversation, summarise_the_conversation, generating_response,prompt_constructor, set_background, sidebar_bg, read_pdf, read_docx
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
#from langchain.docstore.document import Document
from langchain.schema.document import Document



def get_kaggle_recommendation(user_input, llm):  
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.load_local("vectorstore/db_faiss", embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 15})
    custom_prompt_template = """
    
    Imagine that you are a Kaggle Recommendation System.
    Your job is to give the best possible information to the user about Kaggle Datasets in properly structured format.
    Ignore the "Dataset-Nr." from the data.
    Arrange all the datasets in ascending order.
    I want minimum 15 datasets no matter what context data you have (Follow this condition strictly).
    At the end of response, add the text : I hope this was a helpful response. Now you can talk with the recommended data.
    
    Construct the response in the below structure-
    - Sr.No (which needs to begin from 1..)
    - Dataset Name
    - Title
    - Description
    - Author
    - Link
    - Last updated
    - Size
    - Usability rating
    - View count
    - Licence
    - Tags
    ----------------------------------------------------------------
    - Sr.No 
    - Dataset Name
    ...etc
    .......
    
    If you do not have any information, you can rerspond with : Sorry, I don't have any information about the datasets.
    
    Below is the context :
    {context}
    
    This is the extract of the text for which you need find datasets: 
    {question}
    """
    # Create a PromptTemplate instance with your custom template
    custom_prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )
    # Use your custom prompt when creating the ConversationalRetrievalChain
    
    memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
     )
    
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        verbose=True,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )        
    response = qa({"question": user_input})['answer']
    return response



