import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
import markdown2
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# --- CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # langchain-pinecone uses this automatically
PINECONE_INDEX_NAME = "ayurveda-rag"

# --- GLOBAL VARIABLES ---
chain = None
retriever = None

# --- LIFESPAN EVENT HANDLER (The New Way) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain, retriever
    print("Application startup: Initializing RAG chain from cloud services...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
        
        # Use the modern langchain-pinecone library's from_existing_index method
        # It handles client initialization internally using environment variables
        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        prompt_template = """
        You are an expert Ayurvedic Dietitian named "AyurMind". Provide clear, friendly, human-readable answers based on the context provided.
        Context: {context}
        Question: {question}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        llm = ChatGroq(model="gemma2-9b-it", temperature=0.3, groq_api_key=GROQ_API_KEY)
        
        chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        
        print("✅ RAG chain loaded successfully.")
    except Exception as e:
        print(f"!!! CRITICAL ERROR DURING STARTUP: Could not initialize RAG chain. {e}")
    
    yield # The application runs here

# --- INITIALIZATION ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PDF GENERATION ---
def generate_pdf(reply_text, filename="ayurveda_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CustomNormal", parent=styles["Normal"], fontSize=12, leading=16))
    story = [Paragraph(line, styles["CustomNormal"]) for line in markdown2.markdown(reply_text).splitlines() if line.strip()]
    doc.build(story)
    return FileResponse(filename, filename=filename, media_type="application/pdf")

# --- DATA MODELS ---
class Query(BaseModel):
    question: str

# --- MEMORY ---
conversation_histories = {}

# --- API ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse("<h3>AyurMind API is running. Connect your frontend to the /ask endpoint.</h3>", status_code=200)

@app.get("/health")
def health_check():
    if chain and retriever:
        return {"status": "ok", "message": "RAG chain is ready."}
    else:
        return {"status": "error", "message": "RAG chain failed to initialize. Check server logs."}

@app.post("/ask")
async def ask_question(query: Query, request: Request):
    if not chain or not retriever:
        return JSONResponse(status_code=503, content={"error": "RAG chain is not ready. The server may be starting or encountered an error."})

    session_id = request.headers.get("X-Session-Id") or str(uuid.uuid4())
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    conversation_history = conversation_histories[session_id]
    conversation_history.append({"role": "user", "content": query.question})
    
    answer = chain.invoke(query.question)
    
    conversation_history.append({"role": "assistant", "content": answer})

    return JSONResponse({
        "mode": "text",
        "answer": answer,
        "session_id": session_id
    })
    
@app.post("/diet-chart/pdf")
async def diet_chart_pdf(payload: dict):
    diet_chart = payload.get("dietChart", "")
    session_id = str(uuid.uuid4())
    filename = f"ayurveda_diet_{session_id}.pdf"
    return generate_pdf(diet_chart, filename)