from langchain_community.chat_models import ChatOpenAI

chat = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",  # Required for OpenRouter
    openai_api_key="",   # Replace with your OpenRouter API key
    model_name="openai/gpt-3.5-turbo",               # Use OpenRouter's model naming
)

response = chat.predict("What is the capital of india")
print(response)
