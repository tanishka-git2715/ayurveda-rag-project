import os
import asyncio
import threading
import uuid
import json
from typing import Optional, List, Dict
from queue import Queue
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
import markdown2
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


# --- PDF GENERATION ---
def generate_pdf(reply_text, filename="ayurveda_report.pdf"):
    doc = SimpleDocTemplate(
        filename, pagesize=A4,
        rightMargin=40, leftMargin=40,
        topMargin=60, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CustomHeading1", parent=styles["Heading1"], fontSize=16, spaceAfter=12))
    styles.add(ParagraphStyle(name="CustomHeading2", parent=styles["Heading2"], fontSize=14, spaceAfter=10))
    styles.add(ParagraphStyle(name="CustomNormal", parent=styles["Normal"], fontSize=12, leading=16))

    story = []
    html_text = markdown2.markdown(reply_text)

    for block in html_text.split("\n"):
        block = block.strip()
        if not block:
            story.append(Spacer(1, 8))
            continue
        story.append(Paragraph(block, styles["CustomNormal"]))

    doc.build(story)
    return FileResponse(filename, filename=filename, media_type="application/pdf")

# --- CONFIG ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

DATA_PATH = "data/"
VECTOR_STORE_PATH = "vectorstore/"

chain = None
retriever = None
rebuild_lock = threading.Lock()

origins = ["http://localhost:5173", "http://127.0.0.1:5173", "https://ayurdev-patient.vercel.app", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

REQUIRED_FIELDS = {
    "diet": ["age", "gender", "height/weight", "activity level", "health conditions", "dietary preferences", "goals"],
    "pdf": ["age", "gender", "body type", "lifestyle", "sleep pattern", "stress level", "health conditions", "goals"]
}

# --- RAG CHAIN ---
def create_rag_chain(retriever):
    prompt_template = """
You are an expert Ayurvedic Dietitian named "AyurMind". Provide clear, friendly, human-readable answers.

Context:
{context}

Question:
{question}
"""
    def groq_model_runner(input_text: str) -> str:
        input_text = str(input_text).strip()
        if not input_text:
            return "No question provided."
        messages = [
            {"role": "system", "content": "You are AyurMind, an expert Ayurvedic Dietitian."},
            {"role": "user", "content": input_text}
        ]
        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=messages,
            temperature=0.3,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | groq_model_runner | StrOutputParser())
    return rag_chain

# --- MEMORY ---
conversation_histories = {}  # session_id -> { messages: [], last_chart_mode: str, last_chart_data: str }

# --- ASK ENDPOINT WITH MULTI-TURN SUPPORT ---
@app.post("/ask")
async def ask_question(query: Query, request: Request):
    global chain, retriever
    if chain is None or retriever is None:
        return {"error": "RAG chain not ready. Run /rebuild first."}

    session_id = request.headers.get("X-Session-Id") or str(uuid.uuid4())
    if session_id not in conversation_histories:
        conversation_histories[session_id] = {"messages": [], "last_chart_mode": None, "last_chart_data": None}

    session_data = conversation_histories[session_id]
    user_input = query.question
    session_data["messages"].append({"role": "user", "content": user_input})

    docs = retriever.get_relevant_documents(user_input)
    context_text = "\n".join([doc.page_content for doc in docs]) or "No relevant context."

    # Mode classification
    classification_prompt = f"""
Classify the query into one of: "text", "diet", "pdf".
Return ONLY the mode.

Query:
{user_input}
"""
    mode_response = await asyncio.to_thread(chain.invoke, classification_prompt)
    mode = mode_response.strip().lower()
    if mode not in ["text", "diet", "pdf"]:
        mode = "text"

    # If follow-up, persist original chart mode
    if mode == "text" and session_data.get("last_chart_mode"):
        mode = session_data["last_chart_mode"]

    # Build task-specific prompt
    if mode in ["diet", "pdf"]:
        session_data["last_chart_mode"] = mode
        missing_fields = ", ".join(REQUIRED_FIELDS[mode])
        role = "Dietitian" if mode == "diet" else "Wellness Consultant"
        output_type = "7-day personalized diet chart in Markdown" if mode == "diet" else "comprehensive wellness report"
        system_prompt = f"""
You are AyurMind, an expert Ayurvedic {role}.

Phased Task:
1. If missing details ({missing_fields}) → ask follow-ups conversationally.
2. If enough info → generate a {output_type}.
3. If some info missing → mark as "N/A" but still provide the result.

Instruction:
- also add heading as final diet chart or final wellness report whatever user asked.

Context: {context_text}
"""
    else:
        system_prompt = f"""
You are AyurMind, an expert Ayurvedic Dietitian.
Context: {context_text}
"""

    # Combine system + history
    messages = [{"role": "system", "content": system_prompt}] + session_data["messages"]

    completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=messages,
        temperature=0.3,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )
    reply = completion.choices[0].message.content
    session_data["messages"].append({"role": "assistant", "content": reply})

    # Track last chart data
    if mode in ["diet", "pdf"]:
        session_data["last_chart_data"] = reply

    # Mode-specific return
    if mode == "pdf":
        return JSONResponse({
            "mode": mode,
            "wellness_report": session_data["last_chart_data"],
            "is_wellness_ready": True
        })
    elif mode == "diet":
        pattern = re.compile(r"(breakfast|lunch|dinner|day\s*1|day\s*2|day\s*3|day\s*4|day\s*5|day\s*6|day\s*7)")
        final_chart = bool(pattern.search(reply.lower()))
        return JSONResponse({
            "mode": mode,
            "diet_chart": session_data["last_chart_data"],
            "is_final_chart": final_chart
        })
    else:
        return JSONResponse({
            "mode": mode,
            "answer": reply,
            "session_id": session_id
        })

# --- PDF ENDPOINT ---
@app.post("/diet-chart/pdf")
async def diet_chart_pdf(payload: dict):
    diet_chart = payload.get("dietChart", "")
    session_id = str(uuid.uuid4())
    filename = f"ayurveda_diet_{session_id}.pdf"
    return generate_pdf(diet_chart, filename)


# --- REBUILD WORKER ---
def rebuild_worker(q: Queue):
    global chain, retriever
    q.put("Starting rebuild...")
    q.put(f"--- Searching for documents in: {os.path.abspath(DATA_PATH)}")
    q.put(f"--- Files found: {os.listdir(DATA_PATH)}")

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
            q.put("No documents found. Aborting.")
            q.put("DONE")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)
        q.put(f"Split into {len(chunks)} chunks.")

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        vector_store.save_local(VECTOR_STORE_PATH)

        db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 5})
        chain = create_rag_chain(retriever)
        q.put("RAG chain rebuilt successfully.")

    except Exception as e:
        q.put(f"Unexpected error: {e}")
    finally:
        q.put("DONE")

@app.get("/rebuild")
async def rebuild_endpoint():
    if not rebuild_lock.acquire(blocking=False):
        return {"error": "Rebuild already in progress."}

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

    headers = {"Content-Type": "text/event-stream"}
    return StreamingResponse(stream_generator(), headers=headers)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse("<h3>AyurMind API is running.</h3>", status_code=200)

@app.get("/health")
def health():
    return {"status": "ok"}

# --- STARTUP ---
@app.on_event("startup")
async def startup_event():
    global chain, retriever
    if os.path.exists(VECTOR_STORE_PATH) and any(os.scandir(VECTOR_STORE_PATH)):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_kwargs={"k": 5})
            chain = create_rag_chain(retriever)
            print("RAG chain loaded successfully.")
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            chain = None
            retriever = None
    else:
        print("No vectorstore found. Run /rebuild first.")
