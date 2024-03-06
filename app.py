import os
import streamlit as st

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter

from utils import get_kaggle_recommendations, download_the_conversation, summarise_the_conversation , read_docx 


def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=40)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


def chat_retriever(response_from_llm):
    print("response_from_llm :", response_from_llm)
    documents = get_text_chunks_langchain(response_from_llm)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 15})
    return retriever

st.set_page_config(
        page_title="KaggleGPT",
        page_icon='🤖',
        layout='centered',
        initial_sidebar_state='collapsed'
    )

st.header("KaggleGPT")
st.subheader("Dataset Recommender System via Large Language Models")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

with st.sidebar:
    if st.button("Summarise conversation"):
        summarise_the_conversation(st.session_state.messages)
        
    if st.button("Download converation"):
        st.success("PDF created successfully! Click the button below to download.")
        # Create PDF and get its filename
        pdf_filename = download_the_conversation(st.session_state.messages)
        # Provide the generated PDF for download
        st.download_button(
        label="Download PDF",
        data=open(pdf_filename, 'rb').read(),
        file_name="conversation.pdf",
        mime='application/pdf')   
         
    if st.button("New Conversation"):
        st.session_state.messages = []
        
        
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
llm = ChatOpenAI(model_name = "gpt-4", streaming=True)


uploaded_file = st.file_uploader("Upload a DOCX file", type=["docx"])

if "is_docx_file_uploaded" not in st.session_state:
    st.session_state.is_docx_file_uploaded = False
              
if uploaded_file is not None:
    st.success("File uploaded successfully!")

    user_input_docx_text = read_docx(uploaded_file)
    st.session_state.is_docx_file_uploaded = True
    uploaded_file = st.empty()
    
actions = ["Profile Based", "Expert Based", "Knowledge Based","Multi-Criteria Based"]
selected_action = st.selectbox("Recommendation Type:", actions) 

if st.session_state.is_docx_file_uploaded and selected_action:
    if "is_recommendation_generated" not in st.session_state:
            st.session_state.is_recommendation_generated = False

    if not st.session_state.is_recommendation_generated:
        with st.spinner("Wait....Fetching kaggle datasets recommendations..."):
            response_from_llm = get_kaggle_recommendations(user_input_docx_text,selected_action,llm)
            st.session_state.response_from_llm = response_from_llm
            st.success("Fetched response:")
            st.markdown("**__Here are the best datasets recommended:__**")
            st.markdown(response_from_llm) 
            st.session_state.messages.append({"role": "user", "content": user_input_docx_text})
            st.session_state.messages.append({"role": "assistant", "content": response_from_llm})
            st.session_state.is_recommendation_generated = True


    if st.session_state.is_recommendation_generated:
        retriever = chat_retriever(st.session_state.response_from_llm)
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = retriever 

def chat_response(user_input):
    custom_prompt_template = """
        Answer all the questions strictly with the below context. If user asked the question out of context, please respond with "Sorry, but I don't know about it.". Please do not give any information other this context :
        {context}
        
        Question: {question}
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
            retriever=st.session_state.retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
        )        
    response = qa({"question": user_input})['answer']
    return response
    
    
    
user_input = st.chat_input("Ask me anything...")



if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Display assistant response in chat message container
    #response = qa({"question": user_input})['answer']
    response = chat_response(user_input)        #qa({"question": user_input})['answer']
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)
