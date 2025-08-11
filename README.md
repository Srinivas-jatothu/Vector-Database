# Retrieval-Augmented Generation with LangChain and Google Gemini

This project demonstrates a simple yet powerful Retrieval-Augmented Generation (RAG) system built using Python, LangChain, and Google's Gemini models. The application can answer questions based on a collection of local text documents, making it a useful tool for creating custom knowledge bases and question-answering bots.

## The Power of Vector Databases

At the heart of this RAG system is a **vector database**, which is a new type of database designed to store and search through high-dimensional data, such as the text embeddings used in this project. Unlike traditional databases that rely on exact keyword matching, vector databases enable **semantic search**, which allows you to find information based on its meaning and context rather than just the words used.

### Why ChromaDB?

This project uses **ChromaDB**, a popular open-source vector database, for several key reasons:

-   **Simplicity**: ChromaDB is easy to set up and use, making it an excellent choice for developers who are new to vector databases.
-   **Performance**: It is highly optimized for fast similarity searches, which is crucial for real-time applications like this one.
-   **Flexibility**: ChromaDB can be run in-memory for quick prototyping or as a persistent, client-server database for larger-scale deployments.

## How It Works

The core idea behind this RAG system is to combine the power of a large language model (LLM) with a local knowledge base. Here's a step-by-step breakdown of the process:

1.  **Document Loading**: The application starts by loading all `.txt` files from a `data/` directory. This is where you place your custom knowledge base.
2.  **Text Chunking**: Since LLMs have a limited context window, the loaded documents are split into smaller, manageable chunks. This ensures that only the most relevant information is passed to the model.
3.  **Embedding and Vector Storage**: Each text chunk is converted into a numerical representation called an "embedding" using Google's `embedding-001` model. These embeddings are then stored in a ChromaDB vector store, which allows for efficient similarity searches.
4.  **Retrieval**: When a user asks a question, the application first converts the query into an embedding. It then uses the vector store to find the most similar text chunks from the knowledge base.
5.  **Generation**: The retrieved text chunks are then passed to the Gemini Pro LLM along with the original question. The LLM uses this context to generate a relevant and accurate answer.

## Getting Started

Follow these instructions to get the application up and running on your local machine.

### Prerequisites

-   Python 3.8 or higher
-   A Google API key with the Gemini API enabled. You can get one from the [Google AI Studio](https://aistudio.google.com/app/apikey).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/Srinivas-jatothu/Vector-Database.git](https://github.com/Srinivas-jatothu/Vector-Database.git)
    cd Vector-Database
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**

    -   Create a `.env` file in the root of the project.
    -   Add your Google API key to the `.env` file:

        ```
        GOOGLE_API_KEY="your-google-api-key"
        ```

### Usage

1.  **Add your documents:**

    -   Create a `data/` directory in the root of the project.
    -   Place all of your `.txt` files inside the `data/` directory.

2.  **Run the application:**

    ```bash
    python your_script_name.py
    ```

The application will then load your documents, create the vector store (if it doesn't already exist), and answer the example question. You can modify the `query1` variable in the script to ask your own questions.
