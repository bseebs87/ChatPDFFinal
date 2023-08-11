from io import BytesIO
import fitz
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import config
from sentence_transformers import SentenceTransformer
import uuid



openai_api_key = config.openai_api_key
pinecone_api_key = config.pinecone_api_key
pinecone_env_key = config.pinecone_environment
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env_key)
index = pinecone.Index('totalcare')
model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')


class PDF:
    def __init__(self, file):
        self.file = file

    
    
    def pdfProcessing(self, file):
        with BytesIO(file.read()) as file_obj:
            # Create a PDF reader object
            doc = fitz.open("pdf", file_obj)
            metadata = doc.metadata
            print(metadata)
        
            # Read the text content of the PDF
            text = ""
            for page in doc:
            
                text += page.get_text()
                

        first_clean = re.sub(r'\n+', ' ', text)  # Remove line breaks
        cleaned_text = re.sub(r'\s+', ' ', first_clean)  # Remove extra space

        sub_str = "references"

        if sub_str in cleaned_text:
            res = cleaned_text[:str.lower(cleaned_text).find(sub_str) + len(sub_str)]
        else: res = cleaned_text


    
        #Calling the text splittin function   
        self.split(res, 1250, metadata)

    
    def split(self, data, chunk_size, meta):
        #Langchain text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=25, separators=[" ", ",", "\n"]
        )
        
        #Calling the tex_splitter wrapper
        chunks = text_splitter.create_documents([data])
        
        #Calling the upload function
        self.pdf_upsert(chunks, meta)    

    def pdf_upsert(self, chunks, metadata):
        ids = []
        pinecone_vectors = []
        vector = []
        metadatas = []
        
        for j,chunk in enumerate(chunks):
            print("\tCreating embedding for chunk ", j + 1, "of ", len(chunks))
            #Adding the embeddings to the vector
            vector.append(self.doc_embeddings(chunk.page_content))
            ids.append(str(uuid.uuid4()))
            if metadata:
                metadatas.append({'text': chunk.page_content, "author": metadata["author"], "title": metadata["title"], "creation_date": metadata['creationDate']})  #Creation date format is YYYMMDDhhmmss (YYYY - year, MM - month, DD - day, hh - hour, mm - minute, ss - second), and <TZ> is a time zone value (time interval relative to GMT) 
                                                                                                                                                                #containing a sign (‘+’ or ‘-‘), the hour (hh), and the minute (‘mm’, note the apostrophes!).
                                                                                                                                                                #TZ> is a time zone value (time interval relative to GMT)
        
            print(metadatas[j])

    
           
            print("\tAdding vector to pinecone_vectors list for chunk ", j + 1, " of ", len(chunks))
           
            
            if len(vector) % 100 == 0:
                #Creation of Vectors
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