from docx import Document
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph
from reportlab.lib.styles import getSampleStyleSheet


import streamlit as st


from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.chains import ConversationChain
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)


def get_custom_prompt_template(recommendation_engine):
    if recommendation_engine == "Profile-Based":
        custom_prompt_template = """
        Project Exposé must be in computer science, machine learning, and artificial intelligence in general. Students might not give a precise description and ideas. They just want to have any proposed topics with datasets.
        Your tasks are as follows:
        1. You should provide at least 10 different datasets on computer vision, natural language processing, or time series.
        2. You should display results in a table for easy viewing.
        3. You should group datasets by topic.
        4. The response should have the Kaggle Datasets link.
        
        This is the extract of the text for which you need to find datasets: 
        {question}
        
        Below is the context :
        {context}
        
        You identify the input language and give the response in the same language, However, you should provide the dataset's information in English.
        
        """
    
    elif recommendation_engine == "Expert-Based": 
        custom_prompt_template = """
        Based on the Project Exposé, KaggleGPT combines current trends and topics in the fields and proposes challenging ideas with datasets. The output here is intended for good students who want to pursue challenging ideas.
        Your tasks are as follows:
        1. You should summarize several current interesting trends to persuade students working on challenging datasets.
        2. You should display results in a table for easy viewing.
        3. You should provide at least 8 different datasets.
        4. You should sort the datasets by size and usabilityRating. The larger the size and usabilityRating, the more difficult to work with those datasets.
        5. You should provide extra advances, such as requiring students to consider using powerful computing systems or cloud platforms to work with large datasets, developing a runnable prototype, or deploying a demo.
        6. The response should have the Kaggle Datasets link.
        
        Below is the context :
        {context}
        This is the extract of the text for which you need to find datasets: 
        {question}
        
        You identify the input language and give the response in the same language, However, you should provide the dataset's information in English.
        """
        
    elif recommendation_engine == "Knowledge-Based":
        custom_prompt_template = """
        The outputs are purely based on the master programs and syllabus with fixed learning outcomes. How a regular project should look like. Your tasks are as follows:
        1. You should provide at least 10 different datasets on computer vision, natural language processing, or time series.
        2. You should display results in a table for easy viewing.
        3. You should group datasets by topic.
        4. You should sort the datasets by viewCount and voteCount. The larger the viewCount and voteCount, the more popular to work with those datasets.
        5. The response should have the Kaggle Datasets link.
        
        Below is the context :
        {context}
        This is the extract of the text for which you need to find datasets: 
        {question}
        
        You identify the input language and give the response in the same language, However, you should provide the dataset's information in English.
        """
        
    elif recommendation_engine == "Multi-Criteria Based":
        custom_prompt_template = """
        The combined recommendation considers other meta information, such as how long is the thesis duration. Is the topic suitable for the restricted time frame? Do students invest in GPU workstation or cloud computing to run experiments? Do students want to have a conference and journal submission out of the results. KaggleGPT might ask the students if they have the required criteria. Your tasks are as follows:
        1. You should summarize several current interesting trends to persuade students working on challenging datasets.
        2. You should display results in a table for easy viewing.
        3. You should provide at least 8 different datasets.
        4. You should sort the datasets by size, usabilityRating, viewCount and voteCount. The larger the numbers, the more challenging to work with those datasets.
        5. The response should have the Kaggle Datasets link.
        6. You should mention that submitting a research paper is highly recommended.
        7. You should provide extra advances, such as requiring students to consider using powerful computing systems or cloud platforms to work with big datasets. Students must also develop a runnable prototype or deploy a demo.
       
        Below is the context :
        {context}
        This is the extract of the text for which you need to find datasets: 
        {question}

        You identify the input language and give the response in the same language. However, you should provide the dataset's information in English.
        """
    return custom_prompt_template
             

def get_llm_response(user_query,open_ai_llm,retriever,memory,recommendation_engine): 
    template = get_custom_prompt_template(recommendation_engine)
    # Create a PromptTemplate instance with your custom template
    custom_prompt = PromptTemplate(
        template= template,
        input_variables=["context", "question"],
    )
    # Use your custom prompt when creating the ConversationalRetrievalChain

    qa = ConversationalRetrievalChain.from_llm(
        open_ai_llm,
        verbose=True,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )        
    response = qa({"question": user_query})['answer']
    return response

def download_the_conversation(messages, filename='conversation.pdf'):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []

    title_style = getSampleStyleSheet()["Title"]
    title_data = [
        Paragraph('<b><u>KaggleGPT : Prompt-based Recommender System for Efficient Dataset Discovery</u></b>', title_style),
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

def read_pdf(file):
    doc = PdfReader(file)
    text = ""
    for page in doc.pages:
        text += page.extract_text()
    return text


def initial_response_query_and_answer(query, bots_response,recommendation_engine):
    response_text = f"You are a {recommendation_engine} Kaggle Dataset Recommendation Engine \n user query: {query}\n\nrecommeded datasets information: {bots_response}"
    return response_text

def conversation_object(recommendation_engine):  
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3,return_messages=True)
        
    if recommendation_engine == "Profile-Based":
        template = """
        You are a dataset recommendation system that gives students the required datasets and answers all possible questions based on the context and history of the chat. The students are master students in machine learning, data science, and artificial intelligence. 
        
        You identify the input language of user from context and give the response in the same language. 
        
        Here is the context:"""
        
    elif recommendation_engine == "Expert-Based":
        template = """
        You are a dataset recommendation system that gives students the required datasets and answers all possible questions based on the context and history of the chat. You combine your latest knowledge with the context and provide challenging datasets. The challenging datasets are defined by data size and usabilityRating. The larger the size and usabilityRating, the more challenging the datasets. Please provide external datasets if needed. 
        
        You identify the input language of user from context and give the response in the same language. 
        
        Here is the context:
        """
    
    elif recommendation_engine == "Knowledge-Based":
        template = """
        You are a dataset recommendation system that gives students the required datasets and answers all possible questions based on the chat's context and history. The students are master's students in machine learning, data science, and artificial intelligence. They are in their last year and studying the necessary prerequisite courses. You provide datasets based on viewCount and voteCount information. The larger the viewCount and voteCount, the more popular it is to work with those datasets. 
        
        You identify the input language of user from context and give the response in the same language. 
        
        Here is the context:
        """
    
    elif recommendation_engine == "Multi-Criteria Based":
        template = """
        You are a dataset recommendation system that gives students the required datasets and answers all possible questions based on the context and history of the chat. You combine with your latest knowledge, align with the context, and provide challenging datasets focused on size, usabilityRating, viewCount and voteCount. The larger the numbers, the more difficult it is to work with those datasets. Please provide external datasets if needed. It would help if you mentioned that publication is a must after experiments. It would help if you gave external baselines based on your latest knowledge. 
        
        You identify the input language of user from context and give the response in the same language. 
        
        Here is the context:
        """
          
    system_msg_template = SystemMessagePromptTemplate.from_template(template=template)
    
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
    conversation = ConversationChain(memory=st.session_state.buffer_memory,   prompt=prompt_template, llm=st.session_state.open_ai_llm, verbose=True)
    
    return conversation

