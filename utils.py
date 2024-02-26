from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


def summarise_the_conversation():
    pass

def download_the_conversation():
    pass

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

