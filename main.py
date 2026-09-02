import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=SecretStr(api_key),
    stop_sequences=[],
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant. Answer the user's question clearly and briefly.",
        ),
        ("human", "{question}"),
    ]
)

query = input("Enter your research query: ")
response = (prompt | llm).invoke({"question": query})
print(response.content)
