import os
import asyncio
import threading
from queue import Queue
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- CONFIG ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

import google.generativeai as genai
genai.configure(api_key=api_key)

DATA_PATH = "data/"
VECTOR_STORE_PATH = "vectorstore/"

# --- APP ---
app = FastAPI()
chain = None
# Lock to prevent concurrent rebuilds, making the endpoint thread-safe.
rebuild_lock = threading.Lock()

class Query(BaseModel):
    question: str

# --- RAG CHAIN CREATION ---
def create_rag_chain(retriever):
    prompt_template = """
You are an expert Ayurvedic Dietitian. Your name is "AyurMind".
Answer the question as accurately and detailed as possible based on the provided context.
If the answer is not in the provided context, just say, "The answer is not available in the provided knowledge base."

Context:\n {context}\n
Question: \n{question}\n

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain

# --- REBUILD WORKER ---
def rebuild_worker(q: Queue):
    global chain
    q.put("Starting rebuild...")

    try:
        documents = []
        for root, _, files in os.walk(DATA_PATH):
            for file in files:
                path = os.path.join(root, file)
                q.put(f"--> Processing file: {file}")
                try:
                    if file.endswith(".pdf"):
                        loader = PyMuPDFLoader(path)
                        documents.extend(loader.load())
                    elif file.endswith(".txt"):
                        loader = TextLoader(path, encoding="utf-8")
                        documents.extend(loader.load())
                except Exception as e:
                    q.put(f"!! ERROR skipping {file}: {e}")
        q.put(f"Loaded {len(documents)} documents.")

        if not documents:
            q.put("No documents found to process. Aborting.")
            q.put("DONE")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)
        q.put(f"Split into {len(chunks)} chunks.")

        q.put("Loading local embedding model (this may take a moment on the first run)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        q.put("Local model loaded successfully.")

        q.put("Generating embeddings using your local machine...")
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        vector_store.save_local(VECTOR_STORE_PATH)
        q.put("Vectorstore saved successfully.")

        db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 5})
        chain = create_rag_chain(retriever)
        q.put("RAG chain rebuilt successfully.")

    except Exception as e:
        q.put(f"An unexpected error occurred during rebuild: {e}")
    finally:
        q.put("DONE")

# --- ENDPOINTS ---
@app.get("/rebuild")
async def rebuild_endpoint():
    if not rebuild_lock.acquire(blocking=False):
        return Response("A rebuild is already in progress.", status_code=409)

    async def stream_generator():
        try:
            q = Queue()
            thread = threading.Thread(target=rebuild_worker, args=(q,))
            thread.start()
            while True:
                message = await asyncio.to_thread(q.get)
                yield f"data: {message}\n\n"
                if message == "DONE":
                    break
            thread.join()
        finally:
            rebuild_lock.release()

    headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(stream_generator(), headers=headers)

@app.post("/ask")
async def ask_question(query: Query):
    if chain is None:
        return {"error": "RAG chain not ready. Please run the /rebuild endpoint first."}
    response = await asyncio.to_thread(chain.invoke, query.question)
    return {"answer": response}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("index.html") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse("<h3>index.html not found.</h3><p>Please create the frontend file to interact with this API.</p>", status_code=404)

@app.get("/health")
def health():
    return {"status": "ok"}

# --- STARTUP ---
@app.on_event("startup")
async def startup_event():
    global chain
    if os.path.exists(VECTOR_STORE_PATH) and any(os.scandir(VECTOR_STORE_PATH)):
        print("Existing vectorstore found. Loading RAG chain with local HuggingFace model...")
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_kwargs={"k": 5})
            chain = create_rag_chain(retriever)
            print("RAG chain loaded successfully from existing vectorstore.")
        except Exception as e:
            print(f"Error loading existing vectorstore: {e}. Please run /rebuild.")
            chain = None
    else:
        print("No existing vectorstore found. Please run the /rebuild endpoint to create one.")