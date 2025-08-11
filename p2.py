import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

print("Environment setup complete. API key loaded.")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    print("Google Gemini LLM initialized successfully.")
except Exception as e:
    print(f"Error initializing Google Gemini LLM: {e}")
    exit()

# --- 3. DEFINE THE QUERY AND ASK THE QUESTION ---
# Define the question you want to ask.
query = "What is the capital of India?"
print(f"\nAsking the LLM: '{query}'")

# Use the .invoke() method to send the query to the LLM and get the response.
try:
    response = llm.invoke(query)
    print("\n--- Answer ---")
    print(response.content)

except Exception as e:
    print(f"\nAn error occurred while invoking the LLM: {e}")

