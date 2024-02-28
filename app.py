import streamlit as st
from openai import OpenAI
from utils import download_the_conversation, summarise_the_conversation, generating_response,prompt_constructor, set_background, sidebar_bg
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

        
def get_kaggle_recommendations(retriever):
        # Create QA_CHAIN_PROMPT from the template
    user_input_query = st.text_area("Write the topic for which you want Kaggle Datasets recommendations for?", height=100)
    
    actions = ["Profile Based", "Expert Based", "Knowledge Based","Multi-Criteria Based"]
    selected_action = st.selectbox("Recommendation Type:", actions)

    # Button to trigger recommendation
    if st.button("Get Recommendations"):
        if not user_input_query:
            st.error("Please provide a topic for Kaggle Datasets recommendations.")
        else:
            with st.spinner("Wait.Fetching kaggle datasets recommendations..."):
                QUERY_TEMPLATE = prompt_constructor(selected_action)
                print("Query template :")
                print("-"*200)
                print(QUERY_TEMPLATE)
                string_response = generating_response(QUERY_TEMPLATE, retriever, user_input_query)

                # Display the assistant response
                st.success("Fetched response:")
                st.write(string_response)

                # Add user and assistant messages to chat history
                st.session_state.messages.append({"role": "user", "content": user_input_query})
                st.session_state.messages.append({"role": "assistant", "content": string_response})

                # Set flag to indicate that recommendation is generated
                st.session_state.is_recommendation_generated = True

st.set_page_config(
        page_title="KaggleGPT",
        page_icon='🤖',
        layout='centered',
        initial_sidebar_state='collapsed'
    )

st.header("KaggleGPT: Dataset Recommender System via Large Language Models")

set_background('streamlit_ui/img/kagglegpt_background_img.webp')

sidebar_bg('streamlit_ui/img/website.jpg')
# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        mime='application/pdf'
    )   
         
    if st.button("New Conversation"):
        st.session_state.messages = []
   
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
#prompt = st.chat_input("What is up?")
embeddings = OpenAIEmbeddings()
loaded_db = FAISS.load_local("vectorstore/db_faiss", embeddings)
retriever = loaded_db.as_retriever(search_kwargs={"k": 15})

response = False

if "is_recommendation_generated" not in st.session_state:
        st.session_state.is_recommendation_generated = False

if not st.session_state.is_recommendation_generated:
    get_kaggle_recommendations(retriever)
               
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
        print("All messages :",type(st.session_state.messages))   