import os
from dotenv import load_dotenv
import streamlit as st

# LangChain / Google Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

# Load .env (do not hardcode your API key in code in production)
load_dotenv()

# NOTE: prefer using env var rather than hardcoding.

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.warning("GOOGLE_API_KEY not found in environment. Please add to a .env file or export it.")
    
st.set_page_config(page_title="Q&A Demo")
st.title("LangChain + Gemini Q&A (Streamlit)")

# Utility to create the LLM instance. Use client_options and transport="rest" to make explicit use of API key.
def create_llm():
    # client_options with api_key is more reliable in many setups
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.6,
        client_options={"api_key": API_KEY},
        transport="rest",
    )

# Robust extractor for whatever the llm() returns in your LangChain version
def extract_text(resp):
    # If it's already a string
    if isinstance(resp, str):
        return resp
    # LangChain sometimes returns an AIMessage-like object with .content
    if hasattr(resp, "content"):
        return resp.content
    # Some versions return a ChatResult with .generations (list of list)
    if hasattr(resp, "generations"):
        try:
            return resp.generations[0][0].text
        except Exception:
            pass
    # Fallback to str()
    return str(resp)

# UI inputs
prompt = st.text_input("Ask a question", key="input")
submit = st.button("Ask the question")

if submit:
    if not prompt or not prompt.strip():
        st.warning("Please enter a question before submitting.")
    elif not API_KEY:
        st.error("Missing GOOGLE_API_KEY. Add it to your environment or .env and restart the app.")
    else:
        with st.spinner("Contacting Gemini..."):
            llm = create_llm()

            # Use explicit chat messages
            messages = [
                SystemMessage(content="You are a helpful assistant that helps people find information."),
                HumanMessage(content=prompt),
            ]

            # Direct call (most models accept this format)
            raw = llm(messages)

            # Extract and display
            text = extract_text(raw)
            st.subheader("Response")
            st.write(text)




