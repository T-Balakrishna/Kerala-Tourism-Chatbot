import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# ──────────────────────────────────────────────
# Page config (makes it look professional)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Kerala Tourism Chatbot",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/kerala-tourism-chatbot',
        'Report a bug': "mailto:your.email@example.com",
        'About': "Built for Inarva Solutions Internship Assessment – local RAG version"
    }
)

# ──────────────────────────────────────────────
# Sidebar (adds polish & context)
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Flag_of_Kerala.svg/2560px-Flag_of_Kerala.svg.png", width=120)
    st.title("Kerala Tourism Guide")
    st.markdown("**Personalized trip recommendations** using local RAG")
    st.markdown("""
    - Group type: boys, girls, family, college, couples  
    - Season: winter, monsoon, summer  
    - Budget & days: tailored suggestions  
    """)
    st.info("Powered by Ollama (phi3:mini) + LangChain + Chroma\nData from curated CSV (bonus)")
    st.markdown("---")
    st.caption("Local implementation due to Azure tenant/subscription issues during setup")

# ──────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────
st.title("🏝️ Kerala Tourism Chatbot")
st.markdown("Ask anything about planning your perfect Kerala trip — get accurate, personalized suggestions!")

# Quick example buttons (great for demo & video)
st.markdown("### Try these examples")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Boys trip – adventure – winter – ₹15k – 4 days"):
        example = "Best places for boys trip adventure in winter under 15000 for 4 days"
        st.session_state.messages.append({"role": "user", "content": example})
with col2:
    if st.button("Family trip – monsoon – safe & fun"):
        example = "Safe family trip in Kerala during monsoon season"
        st.session_state.messages.append({"role": "user", "content": example})
with col3:
    if st.button("Girls getaway – beach – relaxed"):
        example = "Best beach spots for girls trip in Kerala"
        st.session_state.messages.append({"role": "user", "content": example})

# ──────────────────────────────────────────────
# RAG Logic (unchanged – your original code)
# ──────────────────────────────────────────────
DATA_PATH = "data/kerala_spots.csv"
CHROMA_PATH = "chroma_db"

@st.cache_resource
def load_vectorstore():
    if os.path.exists(CHROMA_PATH):
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
    
    loader = CSVLoader(file_path=DATA_PATH)
    docs = loader.load()
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory=CHROMA_PATH
    )
    return vectorstore

llm = OllamaLLM(model="phi3:mini", temperature=0.6)

prompt = PromptTemplate.from_template(
    """You are a helpful Kerala tourism expert.
Use only the provided context to give accurate, personalized answers.
Include group type, season, budget, days where relevant.
If no relevant info, say so.

Context: {context}

Question: {question}

Answer:"""
)

retriever = load_vectorstore().as_retriever(search_kwargs={"k": 4})

chain = (
    {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ──────────────────────────────────────────────
# Chat interface
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your local Kerala travel assistant. Ask me about group type, budget, season, days — I'll give you personalized suggestions! 😊"}
    ]

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message["content"])

# Input box
if prompt_input := st.chat_input("Ask about your Kerala trip (e.g., boys trip winter budget 15000 4 days)..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Generating personalized recommendation..."):
            response = chain.invoke(prompt_input)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")