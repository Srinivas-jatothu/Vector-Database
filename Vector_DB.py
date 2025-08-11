# import os
# from dotenv import load_dotenv
# import chromadb

# # LangChain components
# from langchain.chains import RetrievalQA
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# # --- FIX ---
# # Import the correct classes for Google's Generative AI models
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


# # --- 1. SET UP THE ENVIRONMENT ---
# # Load environment variables from the .env file
# load_dotenv()

# # --- FIX ---
# # Check if the Google API key is set
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

# # Define the path for the data directory and the persistent ChromaDB storage
# DATA_PATH = "data/"
# PERSIST_DIRECTORY = "db"

# print("Environment setup complete.")



# # --- 2. LOAD AND CHUNK THE DOCUMENTS ---
# print("Loading documents...")
# # Use DirectoryLoader to load all .txt files from the specified directory
# # It uses TextLoader for each file by default.
# loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
# documents = loader.load()

# if not documents:
#     raise ValueError(f"No documents found in the '{DATA_PATH}' directory.")

# print(f"Loaded {len(documents)} documents.")

# # Chunk the loaded documents into smaller pieces for processing
# print("Chunking documents...")
# # RecursiveCharacterTextSplitter is good for keeping related text together.
# # We define a chunk size and an overlap to maintain context between chunks.
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(documents)

# print(f"Split documents into {len(texts)} chunks.")


# # --- 3. INITIALIZE EMBEDDINGS AND VECTORSTORE ---
# print("Checking for existing ChromaDB...")

# # --- FIX ---
# # Initialize the Google embeddings model. This will be used to convert text chunks
# # into numerical vectors (embeddings).
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# print(f"Embeddings initialized with model: {embeddings.model}")




# # Check if the database directory already exists
# if os.path.exists(PERSIST_DIRECTORY):
#     # If it exists, load the existing database from disk
#     print("Loading existing ChromaDB...")
#     vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
#     print("ChromaDB loaded.")
# else:
#     # If it does not exist, create a new one from the documents
#     print("Creating new ChromaDB and storing embeddings...")
#     vectordb = Chroma.from_documents(
#         documents=texts,
#         embedding=embeddings,
#         persist_directory=PERSIST_DIRECTORY
#     )
#     print("ChromaDB has been created and persisted.")

# # --- 4. SETUP THE QUESTION-ANSWERING CHAIN ---
# print("Setting up the RetrievalQA chain...")

# # A retriever is an interface that fetches relevant documents based on a query.
# retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# # --- FIX ---
# # Initialize the Language Model (LLM). We're using Google's Gemini model here.
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

# # Create the RetrievalQA chain.
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True
# )

# print("Chain setup complete. Ready to answer questions.")

# # --- 5. ASK A QUESTION ---
# def process_llm_response(llm_response):
#     """Helper function to print the response and its sources beautifully."""
#     print("\n--- Answer ---")
#     print(llm_response['result'])
#     print("\n--- Sources ---")
#     for source in llm_response["source_documents"]:
#         print(f"- {source.metadata['source']}")


# # Example Question 1
# query1 = "What is the HNSW algorithm and how is it used in vector databases?"
# print(f"\nQuerying the system with: '{query1}'")
# llm_response1 = qa_chain.invoke({"query": query1})
# process_llm_response(llm_response1)






import os
from dotenv import load_dotenv
import chromadb

# LangChain components
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# --- FIX ---
# Import the correct classes for Google's Generative AI models
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


# --- 1. SET UP THE ENVIRONMENT ---
# Load environment variables from the .env file
load_dotenv()

# --- FIX ---
# Check if the Google API key is set
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

# Define the path for the data directory and the persistent ChromaDB storage
DATA_PATH = "data/"
PERSIST_DIRECTORY = "db"

print("Environment setup complete.")



# --- 2. LOAD AND CHUNK THE DOCUMENTS ---
print("Loading documents...")
# Use DirectoryLoader to load all .txt files from the specified directory
# It uses TextLoader for each file by default.
loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

if not documents:
    raise ValueError(f"No documents found in the '{DATA_PATH}' directory.")

print(f"Loaded {len(documents)} documents.")

# Chunk the loaded documents into smaller pieces for processing
print("Chunking documents...")
# RecursiveCharacterTextSplitter is good for keeping related text together.
# We define a chunk size and an overlap to maintain context between chunks.
chunk_size = 1000  # Define chunk size
chunk_overlap = 200  # Define chunk overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
texts = text_splitter.split_documents(documents)

print(f"Split documents into {len(texts)} chunks.")
print(f"Chunk size: {chunk_size} characters")
print(f"Chunk overlap: {chunk_overlap} characters")


# --- 3. INITIALIZE EMBEDDINGS AND VECTORSTORE ---
print("Checking for existing ChromaDB...")

# --- FIX ---
# Initialize the Google embeddings model. This will be used to convert text chunks
# into numerical vectors (embeddings).
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


print(f"Embeddings initialized with model: {embeddings.model}")

# Get the embedding size
example_text = "This is a sample text to get embedding size."
example_embedding = embeddings.embed_query(example_text)
embedding_size = len(example_embedding)

print(f"Embedding vector size: {embedding_size}")


# Check if the database directory already exists
if os.path.exists(PERSIST_DIRECTORY):
    # If it exists, load the existing database from disk
    print("Loading existing ChromaDB...")
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    print("ChromaDB loaded.")
else:
    # If it does not exist, create a new one from the documents
    print("Creating new ChromaDB and storing embeddings...")
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("ChromaDB has been created and persisted.")

# --- 4. SETUP THE QUESTION-ANSWERING CHAIN ---
print("Setting up the RetrievalQA chain...")

# A retriever is an interface that fetches relevant documents based on a query.
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# --- FIX ---
# Initialize the Language Model (LLM). We're using Google's Gemini model here.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

# Create the RetrievalQA chain.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

print("Chain setup complete. Ready to answer questions.")

# --- 5. ASK A QUESTION ---
def process_llm_response(llm_response):
    """Helper function to print the response and its sources beautifully."""
    print("\n--- Answer ---")
    print(llm_response['result'])
    print("\n--- Sources ---")
    for source in llm_response["source_documents"]:
        print(f"- {source.metadata['source']}")


# Example Question 1
query1 = "What is the HNSW algorithm and how is it used in vector databases?"
print(f"\nQuerying the system with: '{query1}'")
llm_response1 = qa_chain.invoke({"query": query1})
process_llm_response(llm_response1)