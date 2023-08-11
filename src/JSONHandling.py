import pinecone
from sentence_transformers import SentenceTransformer
import config
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import json








openai_api_key = config.openai_api_key #Setting up api key
pinecone_api_key = config.pinecone_api_key  #Pinecone API Key
pinecone_env_key = config.pinecone_environment #Pinecone env Key
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env_key)
index = pinecone.Index('totalcare') #test index
model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')  #Embedding model from hugging face

class Json:
    def __init__(self, file):
        self.file = file
#Reading in the JSON file from streamlit, Very poor optimization
    def jsonProcessing(self, file):  
        temp = []
        count = 0

        if file:
            
            
            data = json.load(file)
            print(type(data))
            
            
                   
        print(str(data))
            

            
        

       
        # Split the data into chunks
        self.split(str(data), 1000, []) 

            
            
    
    def split(self, data, chunk_size, meta):
        #Langchain text splitting function
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=25, separators=["}", "\n"]
        )
        
        #Creating the documents for processing
        chunks = text_splitter.create_documents([data])
        print(chunks)

        #calling the uploading function
        self.json_upsert(chunks, meta)    

    def doc_embeddings(self, file):
    
        embeddings = model.encode([file])
        embeddings = embeddings.tolist()  #Convert numpy array to a list
        return embeddings

    
    
    
    def json_upsert(self, chunks, metadata):
        ids = []
        pinecone_vectors = []
        vector = []
        metadatas = []
        
        for j,chunk in enumerate(chunks):
            print("\tCreating embedding for chunk ", j + 1, "of ", len(chunks))
            vector.append(self.doc_embeddings(chunk.page_content))
            ids.append(str(uuid.uuid4()))

            #assigning metadata to the document
            metadatas.append({'text': chunk.page_content})  
        

    
            # add vector to pinecone_vectors list
            print("\tAdding vector to pinecone_vectors list for chunk ", j + 1, " of ", len(chunks))
            #pinecone_vectors.append((ids, vector))
            #print(pinecone_vectors)
            
            if len(vector) % 100 == 0:
                #Upserting for when vector amount reaches 100
                pinecone_vectors = (zip(ids, vector, metadatas))
                print("Upserting batch of 100 vectors...")
                #Pinecone Client upserting wrapper
                upsert_response = index.upsert(vectors=pinecone_vectors)
                pinecone_vectors = []
                ids = []
                vector= []
                metadatas = []
        
        # if there are any vectors left, upsert them
        if len(vector) > 0:
            pinecone_vectors = (zip(ids, vector, metadatas))
            print("Upserting remaining vectors...")
            upsert_response = index.upsert(vectors=pinecone_vectors)
            pinecone_vectors = []
    
            print("Vector upload complete.")
    
