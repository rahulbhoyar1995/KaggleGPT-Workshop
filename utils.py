import base64 
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph
from reportlab.lib.styles import getSampleStyleSheet


def summarise_the_conversation():
    pass

def download_the_conversation(messages, filename='conversation.pdf'):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []

    title_style = getSampleStyleSheet()["Title"]
    title_data = [
        Paragraph('<b><u>Conversation with Kaggle Recommendation System</u></b>', title_style),
        Paragraph('<br/>', title_style)  # Add two break lines before user response
    ]
    story.extend(title_data)


    # Add dialogues to the PDF
    dialogue_style_user = getSampleStyleSheet()["BodyText"]
    dialogue_style_user.fontName = 'Helvetica'  # Change font to Helvetica for better readability
    dialogue_style_assistant = getSampleStyleSheet()["BodyText"]
    dialogue_style_assistant.fontName = 'Helvetica'
    dialogue_style_assistant.alignment = 0  # Align left for assistant's messages

    for message in messages:
        role = message['role']
        content = message['content']

        if role == 'user':
            dialogue_data = [
                Paragraph(f'<b>User:</b> {content}', dialogue_style_user),
                Paragraph('<br/>', dialogue_style_user)  # Add a break line between user and assistant messages
            ]
        else:
            dialogue_data = [
                Paragraph(f'<b>Kaggle Recommender Engine:</b> {content}', dialogue_style_assistant),
                Paragraph('<br/>', dialogue_style_assistant)
            ]

        story.extend(dialogue_data)

    doc.build(story)
    return filename


def run_qa_chain(llm, retriever, QA_CHAIN_PROMPT, question):
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    # Run the query
    result = qa_chain({"query": question})
    
    # Return the result
    return result

def generating_response(query_template, retriever,user_input):
    QA_CHAIN_PROMPT = PromptTemplate.from_template(query_template)

    # Run the QA chain with user input
    llm = ChatOpenAI(model_name = "gpt-4")
    result = run_qa_chain(llm, retriever, QA_CHAIN_PROMPT, user_input)

    # Extract and return the result text
    result_text = result["result"]
    return result_text


def prompt_constructor(kind_of_profile,context ="context",question = "question"):
    QUERY_TEMPLATE = f"""
        Imagine that you are a {kind_of_profile} Kaggle Recommendation System.
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

        Answer the question based only on the following context. Remember that I want the 15 datasets at every cost:
        {'{context}'}
        Question:
        {'{question}'}
        """
    return QUERY_TEMPLATE


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    
    .main {
background-image: url("data:image/png;base64,%s");
background-size: cover;
background-attachment: local;
background-attachment: fixed;
}
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    
    
def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
   
   
 