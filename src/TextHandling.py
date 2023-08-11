from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import config
from sentence_transformers import SentenceTransformer
import uuid
import re

openai_api_key = config.openai_api_key
pinecone_api_key = config.pinecone_api_key
pinecone_env_key = config.pinecone_environment
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env_key)
index = pinecone.Index('totalcare')
model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

class Text:
    def __init__(self, file):
        self.file = file

    #Reading in the text file that was a Bytes io file type
    def textProcessing(self, file):
        string = file.read()
        text = string.decode('utf-8')
        
        print(text)
        # Create a text reader object
        
        
        metadata = file.name
        print(metadata)
        # Read the text content of the PDF
       
       

        first_clean = re.sub(r'\n+', ' ', text)  # Remove line breaks
        cleaned_text = re.sub(r'\s+', ' ', first_clean)  # Remove extra space

        sub_str = "references"

        #Attempting to remove references from certain texts
        if sub_str in cleaned_text:
            res = cleaned_text[:str.lower(cleaned_text).find(sub_str) + len(sub_str)]
        else: res = cleaned_text



        
        #Callling the text splitting function        
        self.split(res, 1250, metadata)

    
    def split(self, data, chunk_size, meta):

         #Splitting the text through Langchain   
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=25, separators=[" ", ",", "\n"]
        )
        
    
        chunks = text_splitter.create_documents([data])
        print(chunks)
        
        #Calling upload function
        self.text_upsert(chunks, meta)    

    def text_upsert(self, chunks, metadata):
        ids = []
        pinecone_vectors = []
        vector = []
        metadatas = []
        
        
        for j,chunk in enumerate(chunks):
            print("\tCreating embedding for chunk ", j + 1, "of ", len(chunks))
            #Calling the embedding function
            vector.append(self.doc_embeddings(chunk.page_content))
            ids.append(str(uuid.uuid4()))
            #Assembling metadatas
            metadatas.append({"text": chunk.page_content, "source": metadata})
           
        
  

    
            # add vector to pinecone_vectors list
            print("\tAdding vector to pinecone_vectors list for chunk ", j + 1, " of ", len(chunks))
            #pinecone_vectors.append((ids, vector))
            #print(pinecone_vectors)
            
            if len(vector) % 100 == 0:
                #Creating the vectors to upload
                pinecone_vectors = (zip(ids, vector, metadatas))
                print("Upserting batch of 100 vectors...")
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
            
    #Embeddings function
    def doc_embeddings(self, file):
    
        embeddings = model.encode([file])
        embeddings = embeddings.tolist()  #Convert numpy array to a list
        return embeddings

    