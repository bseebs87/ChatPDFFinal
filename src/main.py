import streamlit as st
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import os
from langchain.document_loaders import TextLoader, PyPDFLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import LLMChain, RetrievalQA
from pprint import pprint
from langchain import PromptTemplate
import json
import itertools
import pinecone
import itertools
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import configparser
from io import BytesIO
import PyPDF2
import fitz
import re
from transformers import AutoTokenizer
from langchain.vectorstores import Pinecone
import io
import config
import PDFHandling 
import JSONHandling
import TextHandling



# Create a ConfigParser object and read the config.ini file

openai_api_key = config.openai_api_key
pinecone_api_key = config.pinecone_api_key
pinecone_env_key = config.pinecone_environment


pinecone.init(api_key=pinecone_api_key, environment=pinecone_env_key)
index = pinecone.Index('totalcare')
model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
#Prompt Creation using langchain
prompt_template = """

        "Imagine you are an AI assistant for a general practitioner, tasked with improving patient relations. Your job is to answer patient queries about medical conditions, treatments, and general health advice through a patient portal, using data from our comprehensive health database.
        Your responses should maintain a formal tone while ensuring the information is patient-friendly and easily understandable. Each response should be concise, ideally around two paragraphs or less."
        If no context is provided, your description should be "We do not have enough information to give a suggestion" and stop.
        
         
    Query: {query}
    Content: {context}
    Description: """


llm = OpenAI(temperature=.5, openai_api_key=openai_api_key)
#setting the prompt instance
PROMPT = PromptTemplate(template = prompt_template, input_variables = ["query", "context"])
chain = LLMChain(llm=llm, prompt=PROMPT)


def generate_response(query):  #Needs to be 
    
    text = ""
    #Creating an embedding of the query
    embed_query = model.encode([query])
    embed_query = embed_query.tolist() 
    #Pinecone Similarity search 
    results = index.query(
        vector = embed_query, 
        top_k=5, #Returns 4 results
        include_metadata=True
    
    )
    

    print(results)
    #Creating the content to send to an llm if the similarity score is greater than .5
    text = str([x['metadata']['text'] for x in results['matches'] if x['score'] > 0.5])
    print(text)
    


   
    
    inputs = [{text, query}]
    inputs = [{"context": text, "query": query}]
    
    #Chain could be messed with for hallucinations
    return chain.apply(inputs)



# Set the title of the app
st.title('File Upload and Database Query Application')

# Add a file uploader
st.header('Upload your file')

with st.form("my-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a PDF/Text/Json file", type=["txt","json","pdf"] , accept_multiple_files=True)
        submitted = st.form_submit_button(label = "UPLOAD!")




# If a file is uploaded, show a message and offer to process the file

for files in uploaded_file:
        file_extension = os.path.splitext(files.name)[1]
        print(type(files.name))
        
        
        if file_extension == '.txt':
            obj = TextHandling.Text(files)
            obj.textProcessing(files)
           
            
        elif file_extension == ".pdf":
            obj = PDFHandling.PDF(files)
            obj.pdfProcessing(files)
            
            

        elif file_extension == ".json":
            obj = JSONHandling.Json(files)
            obj.jsonProcessing(files)
    
            

    
        else:
            st.error("That type of file is not supported")

        
        st.success("File uploaded successfully!")
        
            
            




    
    
        # Upload the file to the server and get the response code
       
        

        
        # Add code here to process the uploaded file
st.write("Replace with your actual file processing code and display the result here.")





if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Enter your medical question in the text bar below"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ["Welcome to our chatroom"]

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.chat_input("" , key = "input")
    return input_text
    


## Applying the user input box
with input_container:
    user_input = get_text()
    



## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)[0]['text']
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user') 
            message(st.session_state["generated"][i], key=str(i))
