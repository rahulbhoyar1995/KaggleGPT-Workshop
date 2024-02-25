from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def summarise_the_conversation():
    pass

def download_the_conversation():
    pass



from langchain.prompts import PromptTemplate
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from nltk.tokenize import word_tokenize
from langchain.chains import RetrievalQA

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

def generating_response(user_input, retriever):
    # Replace with your actual query template
    QUERY_TEMPLATE = """
        Imagine that you are a Kaggle Recommendation System.
        Your job is to give the best possible information to the user about Kaggle Datasets in properly structured format.
        Ignore the "Dataset-Nr." from the data.
        Arrange all the datasets in ascending order.
        Try to go through the information from all datasets and return with all possible datasets. 
        I want minimum 15 datasets no matter what context data you have (Follow this condition strictly).
        At the end of response, add the text : I hope this was a helpful response. Now you can talk with the recommended data.

        Use the below format-
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

        Answer the question based only on the following context. Remember that I want the 15 datasets at every cost:
        {context}
        Question:
        {question}
        """

    # Create QA_CHAIN_PROMPT from the template
    QA_CHAIN_PROMPT = PromptTemplate.from_template(QUERY_TEMPLATE)

    # Run the QA chain with user input
    llm = ChatOpenAI(model_name = "gpt-4")
    result = run_qa_chain(llm, retriever, QA_CHAIN_PROMPT, user_input)

    # Extract and return the result text
    result_text = result["result"]
    return result_text

# def generating_response(user_input,retriever):
#     template = """
#     Answer the question based only on the following context:
#     {context}
#     Question: {question}
#     """
#     def format_docs(docs):
#         return "\n\n".join([d.page_content for d in docs])

#     prompt = ChatPromptTemplate.from_template(template)
#     model = ChatOpenAI()
   
#     chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | model
#     | StrOutputParser()
#      )
    
#     QUERY = f"""
#     Give me list of all datasets by understanding the below information : 
#     {user_input}

#     I want to know everything about the dataset.
#     Create a proper structure of all the infromation.


#     Igonre the below info :
#     - Has Creator Name
#     - Has Creator URL
#     - Has Current Version Number
#     - Has Description
#     - Has License Name
#     - Has Owner Name
#     - Has Owner Ref
#     - Has Subtitle
#     - Has Title
#     - Has Total Bytes
#     - Has URL
    
#     Here are the few examples of the format I am expecting :
    
#     1. **Dataset Name:** CISI (a dataset for Information Retrieval)
#     - **Tags:** Universities and colleges, Earth and nature
#     - **Creator Name:** HJMason
#     - **Current Version Number:** 1
#     - **Description:** A public dataset from the University of Glasgow's Information Retrieval Group
#     - **Download Count:** 2349
#     - **Files:** None
#     - **License Name:** Other (specified in description)
#     - **Owner Name:** HJMason
#     - **Owner Ref:** dmaso01dsta
#     - **Size:** 759KB
#     - **Subtitle:** A public dataset from the University of Glasgow's Information Retrieval Group
#     - **Topic Count:** 0
#     - **Usability Rating:** 1.0
#     - **View Count:** 29021
#     - **Vote Count:** 24
#     - **URL:** [Link](https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval)

#     2. **Dataset Name:** University Statistics
#     - **Tags:** Universities and colleges
#     - **Creator Name:** Christopher Lambert
#     - **Current Version Number:** 1
#     - **Description:** Statistics surrounding 311 US Universities
#     - **Download Count:** 3559
#     - **Files:** None
#     - **License Name:** CC0: Public Domain
#     - **Owner Name:** Christopher Lambert
#     - **Owner Ref:** theriley106
#     - **Size:** 33KB
#     - **Subtitle:** Statistics surrounding 311 US Universities
#     - **Topic Count:** 0
#     - **Usability Rating:** 0.6875
#     - **View Count:** 32587
#     - **Vote Count:** 70
#     - **URL:** [Link](https://www.kaggle.com/datasets/theriley106/university-statistics)

#     3. **Dataset Name:** University Dataset
#     - **Tags:** Universities and colleges, Education, Exploratory data analysis, Data visualization
#     - **Creator Name:** Anant Prakash Awasthi
#     - **Current Version Number:** 7
#     - **Description:** A fictional dataset, Just remember your college days and start exploring.
#     - **Download Count:** 8250
#     - **Files:** None
#     - **License Name:** Data files © Original Authors
#     - **Owner Name:** Anant Prakash Awasthi
#     - **Owner Ref:** ananta
#     - **Size:** 9MB
#     - **Subtitle:** A fictional dataset, Just remember your college days and start exploring.
#     - **Topic Count:** 0
#     - **Usability Rating:** 0.9705882
#     - **View Count:** 73230
#     - **Vote Count:** 98
#     - **URL:** [Link](https://www.kaggle.com/datasets/ananta/student-performance-dataset)
        

#     """
#     print("Query: ", QUERY)
#     output_str = chain.invoke(QUERY)
#     return output_str



# Give me list of all datasets by understanding the below information : {prompt}

#     I want to know everything about the dataset.
#     Create a proper structure of all the infromation.
    
#     It should be in the below format:
#     Dataset Name: 
#     Title: 
#     Tags: 
#     Keyword:
#     Description: 
#     Subtitle:
#     URL:
#     Creator Name: 
#     Current Version Number: 
#     Download Count: 
#     Files: 
#     License Name: 
#     Owner Name: 
#     Owner Ref: 
#     Size: 
#     Topic Count: 
#     Total Bytes:
#     Usability Rating: 
#     URL: 
#     Kernel Count: 
#     Last Updated: 
#     Versions: 
#     View Count: 
#     Vote Count: 