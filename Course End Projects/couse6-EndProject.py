# Build a chatbot using OpenAI's GPT model + Gradio that can intelligently respond to user queries based on Nestlé's HR policy document.

#pip install openai langchain chromadb pypdf gradio tiktoken

api_key = "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"  # Replace

import os
os.environ["OPENAI_API_KEY"] = api_key


from langchain.document_loaders import PyPDFLoader

# Load the HR policy document
loader = PyPDFLoader("hr-policy-en.pdf")
pages = loader.load_and_split()

# Generate Embeddings and Store in Chroma Vector DB
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Create vector database
embedding = OpenAIEmbeddings()
db = Chroma.from_documents(pages, embedding, persist_directory="./nestle_db") # Specify the directory to persist the vector store
# Persist the vector store
db.persist()

# Load the vector store from the persisted directory
db = Chroma(persist_directory="./nestle_db", embedding_function=embedding)

# Print the number of documents in the vector store
print(f"Vector store contains {len(db)} documents.")    

#Create a Retriever QA Chain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Use GPT-3.5 Turbo as the language model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
retriever = db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
#Build Gradio Chat Interface
import gradio as gr

def chat(query):
    response = qa_chain({"query": query})
    return response["result"]

# Launch Gradio interface
# Create a Gradio interface for the chatbot
interface = gr.Interface(fn=chat, inputs="text", outputs="text",
                         title="Nestlé HR Assistant",
                         description="Ask questions about Nestlé's HR Policy")
interface.launch()

# Observation:
# The chatbot can intelligently respond to user queries based on the HR policy document.
# Check if the chatbot can answer questions like:
# "What is the policy on remote work?" or "How does Nestlé handle employee grievances?"









