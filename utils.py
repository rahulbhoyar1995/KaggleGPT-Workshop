from docx import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph
from reportlab.lib.styles import getSampleStyleSheet


def get_kaggle_recommendations(user_input,selected_action,llm):  
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


def read_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text



