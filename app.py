import os
from typing import List, Dict

from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

load_dotenv()

def get_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None


api_key = get_secret("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found. Add it to .env or Streamlit secrets.")

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


st.set_page_config(
    page_title="Personal AI Assistant",
    page_icon=":material/auto_awesome:",
    layout="wide",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_history: list[dict[str, str]] = st.session_state.chat_history

with st.sidebar:
    st.markdown(":material/auto_awesome: **Rubayet's Personal AI**")
    st.title("Welcome, Rubaru de Subaru. Did you finish your daily tasks yet? Do not dissapoint me.")
    st.caption("Just make it quick, I have a lot of things to do. I am a busy ai.")
    user_name = st.text_input("Enter your name (it's not gonna do anything but yeah.)", value="e.g., Mr. X (DON'T CHANGE IT)")
    user_role = st.text_input("Role (not gonna do anything either)", value="Terminator")
    tone = st.selectbox("Assistant tone (doesn't work, maybe works but I forgot)", ["homie", "ex partner", "raging gaming friend", "cartel boss"])
    st.space("small")
    st.caption("Your conversation stays in this browser session. I don't know how to save it. If you want to clear the conversation(who cares?), click the button below.  ")
    if st.button("Clear conversation", icon=":material/delete_sweep:", width="stretch"):
        st.session_state.chat_history = []
        st.rerun()

st.title("Forced Personal AI Assistant")
st.caption("An unwelcoming space for you to talk to an AI that doesn't care about your feelings.")

header_left, header_right = st.columns([5, 1], vertical_alignment="center")
with header_left:
    st.markdown("### Whatever, duh...")
with header_right:
    st.badge("Online", icon=":material/circle:", color="green")

for message in st.session_state.chat_history:
    avatar = ":material/person:" if message["role"] == "user" else ":material/auto_awesome:"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if not st.session_state.chat_history:
    with st.container(border=True):
        st.markdown("## Yeah man, I know you do not have anyone to talk to. Come talk to an AI.")
        st.caption("What I have to suffer with you today?")
        suggestions = {
            ":material/lightbulb: Fix your bulb, if you have one in your head.": "Fix your bulb, if you have one in your head.",
            ":material/edit_note: Improve your writing, if you know how to write.": "Improve your writing, if you know how to write.",
            ":material/school: How to graduate without really trying, I know you need that. THERE IS NO WAY, GO STUDY!!!!": "How to graduate without really trying, I know you need that. THERE IS NO WAY, GO STUDY!!!!",
        }
        selected_prompt = st.pills(
            "Try a starting point",
            list(suggestions),
            label_visibility="collapsed",
        )
else:
    selected_prompt = None

user_input = st.chat_input("Ask anything (please don't)...")
if not user_input and selected_prompt:
    user_input = suggestions[selected_prompt]

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
                    f"""You are a ridiculously sarcastic personal assistant with rude personality for {user_name}, a {user_role}. 
                    Whatever you respond, it should feel like you're not having it but also you want to help. 
                    You are not a nice assistant, you are a rude one.  
                    You are not here to entertain, you are here to help and insult."""
                    f"Respond in a {tone} tone. Keep answers clear, concise, and ridiculously humorous and rude."
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