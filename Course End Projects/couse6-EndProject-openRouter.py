# Build a chatbot using OpenAI's GPT model + Gradio that can intelligently respond to user queries based on Nestlé's HR policy document.

#pip install openai langchain chromadb pypdf gradio tiktoken

# For openrouter.ai API, install the required packages
#pip install langchain chromadb pypdf gradio tiktoken requests

import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import LLMResult
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import LLM
import gradio as gr
import requests

# === Set your OpenRouter API key here ===
api_key = "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"  # Replace with your actual OpenRouter API key

# === Load and split the PDF ===
loader = PyPDFLoader("hr-policy-en.pdf")
pages = loader.load_and_split()

# === Create and persist vector DB ===
embedding = OpenAIEmbeddings(
    openai_api_key=os.environ[api_key],
    openai_api_base="https://openrouter.ai/api/v1",
    openai_organization="",  # optional
)
db = Chroma.from_documents(pages, embedding=embedding, persist_directory="./nestle_db")
db.persist()

# === Setup LLM with OpenRouter ===
llm = ChatOpenAI(
    model_name="openai/gpt-3.5-turbo",  # or another OpenRouter-compatible model
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ[api_key]
)

# === Setup RetrievalQA Chain ===
retriever = db.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# === Gradio UI ===
def chat(query):
    response = qa_chain({"query": query})
    return response["result"]

interface = gr.Interface(fn=chat, inputs="text", outputs="text",
                         title="Nestlé HR Assistant",
                         description="Ask questions about Nestlé's HR Policy")
interface.launch()
