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

def generating_response(user_input,retriever):
    template = """
    Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()
   
    chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
     )
    
    QUERY = f"""
    Give me list of all datasets by understanding the below information : 
    {user_input}

    I want to know everything about the dataset.
    Create a proper structure of all the infromation.


    Igonre the below info :
    - Has Creator Name
    - Has Creator URL
    - Has Current Version Number
    - Has Description
    - Has License Name
    - Has Owner Name
    - Has Owner Ref
    - Has Subtitle
    - Has Title
    - Has Total Bytes
    - Has URL
    
    Here are the few examples of the format I am expecting :
    
    1. **Dataset Name:** CISI (a dataset for Information Retrieval)
    - **Tags:** Universities and colleges, Earth and nature
    - **Creator Name:** HJMason
    - **Current Version Number:** 1
    - **Description:** A public dataset from the University of Glasgow's Information Retrieval Group
    - **Download Count:** 2349
    - **Files:** None
    - **License Name:** Other (specified in description)
    - **Owner Name:** HJMason
    - **Owner Ref:** dmaso01dsta
    - **Size:** 759KB
    - **Subtitle:** A public dataset from the University of Glasgow's Information Retrieval Group
    - **Topic Count:** 0
    - **Usability Rating:** 1.0
    - **View Count:** 29021
    - **Vote Count:** 24
    - **URL:** [Link](https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval)

    2. **Dataset Name:** University Statistics
    - **Tags:** Universities and colleges
    - **Creator Name:** Christopher Lambert
    - **Current Version Number:** 1
    - **Description:** Statistics surrounding 311 US Universities
    - **Download Count:** 3559
    - **Files:** None
    - **License Name:** CC0: Public Domain
    - **Owner Name:** Christopher Lambert
    - **Owner Ref:** theriley106
    - **Size:** 33KB
    - **Subtitle:** Statistics surrounding 311 US Universities
    - **Topic Count:** 0
    - **Usability Rating:** 0.6875
    - **View Count:** 32587
    - **Vote Count:** 70
    - **URL:** [Link](https://www.kaggle.com/datasets/theriley106/university-statistics)

    3. **Dataset Name:** University Dataset
    - **Tags:** Universities and colleges, Education, Exploratory data analysis, Data visualization
    - **Creator Name:** Anant Prakash Awasthi
    - **Current Version Number:** 7
    - **Description:** A fictional dataset, Just remember your college days and start exploring.
    - **Download Count:** 8250
    - **Files:** None
    - **License Name:** Data files © Original Authors
    - **Owner Name:** Anant Prakash Awasthi
    - **Owner Ref:** ananta
    - **Size:** 9MB
    - **Subtitle:** A fictional dataset, Just remember your college days and start exploring.
    - **Topic Count:** 0
    - **Usability Rating:** 0.9705882
    - **View Count:** 73230
    - **Vote Count:** 98
    - **URL:** [Link](https://www.kaggle.com/datasets/ananta/student-performance-dataset)
        

    """
    print("Query: ", QUERY)
    output_str = chain.invoke(QUERY)
    return output_str



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