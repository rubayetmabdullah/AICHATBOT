import os
from typing import List, Dict

from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=SecretStr(api_key),
    stop_sequences=[],
)

def validate_input(prompt_text: str) -> str:
    cleaned = prompt_text.strip()
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    if not cleaned:
        raise ValueError("Please enter a valid message.")
    if len(cleaned) > 10000:
        raise ValueError("Message is too long. Please keep it under 10000 characters.")
    return cleaned


st.set_page_config(page_title=" Rubayet's Personal AI Assistant", page_icon="🤖", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_history: list[dict[str, str]] = st.session_state.chat_history

with st.sidebar:
    st.title("Profile")
    user_name = st.text_input("Your name", value="User")
    user_role = st.text_input("Role", value="student")
    tone = st.selectbox("Assistant tone", ["friendly", "professional", "casual", "concise"])
    st.caption("Your data stays in the browser session and is only used for this chat.")
    if st.button("Clear chat"):
        st.session_state.chat_history = []

st.title("Personal AI Assistant")
st.caption("A secure, UI-friendly chatbot powered by Groq.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask anything...")

if user_input:
    try:
        safe_input = validate_input(user_input)
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": safe_input})
    with st.chat_message("user"):
        st.markdown(safe_input)

    assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    f"You are a helpful personal assistant for {user_name}, a {user_role}. "
                    f"Respond in a {tone} tone. Keep answers clear, concise, and practical."
                ),
            ),
            ("human", "{question}"),
        ]
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = (assistant_prompt | llm).invoke({"question": safe_input})

            content = response.content
            if isinstance(content, str):
                answer = content.strip()
            else:
                answer = "\n".join(
                    block if isinstance(block, str) else block.get("text", "")
                    for block in content
                    if isinstance(block, (str, dict))
                ).strip()

            st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
