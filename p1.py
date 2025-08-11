import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb

# This wrapper is needed to use LangChain embeddings with a native ChromaDB client
from chromadb.utils.embedding_functions import LangChainEmbeddingFunction

# The 'exceptions' module cannot be imported directly like this in recent versions of chromadb.
# We will refactor the code to not need this import.

# Load environment variables from .env file
load_dotenv()

# --- 1. Check for Environment Variables ---
# Ensure the necessary API keys are set in your .env file
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file and add your API key."
    )
# You are using OpenRouter, so you should have a specific key for it in your .env file
if "OPENROUTER_API_KEY" not in os.environ:
    raise ValueError(
        "OPENROUTER_API_KEY not found in environment variables. "
        "Please add it to your .env file for OpenRouter."
    )


# --- 2. Load Documents ---
# Set the path to your data folder
# IMPORTANT: Replace this with the actual path to your 'data' folder
data_directory = r"C:\Users\jsrin\OneDrive\Desktop\Vector-Database\data"

if not os.path.isdir(data_directory):
    raise FileNotFoundError(f"The specified data directory does not exist: {data_directory}")

print(f"Loading documents from: {data_directory}")
# Use DirectoryLoader to load all .txt files
loader = DirectoryLoader(data_directory, glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")


# --- 3. Split Documents into Chunks ---
# Split documents into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"Split documents into {len(texts)} chunks.")


# --- 4. Create ChromaDB Vector Store ---
# Define the directory to persist the database
persist_directory = 'db'

# Create the vector database from the document chunks
print("Initializing ChromaDB client...")
client = chromadb.PersistentClient(path=persist_directory)

# Define the embedding function using OpenAIEmbeddings and OpenRouter
# NOTE: os.getenv() gets a variable from your environment.
# It correctly loads the key named "OPENROUTER_API_KEY" from your .env file.
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Specify the embedding model
    openai_api_key=os.getenv("OPENROUTER_API_KEY"), # Correctly load key from environment
    openai_api_base="https://openrouter.ai/api/v1"
)

# To use a LangChain embedding function with the native chromadb client,
# you must wrap it with LangChainEmbeddingFunction.
chroma_embedding_function = LangChainEmbeddingFunction(embedding_function)

# Use get_or_create_collection to avoid the old try/except block.
# This is a cleaner way to ensure the collection exists.
# It also requires you to pass the embedding function you intend to use.
print("Getting or creating the collection 'my_collection'...")
collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=chroma_embedding_function
)
print("Collection 'my_collection' is ready.")


# Add texts to the collection only if it's empty to avoid duplicates on re-running the script
if collection.count() == 0:
    print(f"Collection is empty. Adding {len(texts)} document chunks.")
    collection.add(
        documents=[doc.page_content for doc in texts],
        ids=[str(i) for i in range(len(texts))]  # Generate unique IDs for each document
    )
    print("Documents added successfully.")
else:
    print(f"Collection already contains {collection.count()} documents. Skipping add.")


print("Vector database script finished.")
