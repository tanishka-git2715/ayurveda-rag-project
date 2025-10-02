import os
import asyncio
import threading
import uuid
from typing import Optional, List
from queue import Queue
from dotenv import load_dotenv
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

class PatientInfo(BaseModel):
    name: str
    age: int
    gender: str
    constitution: str
    primary_health_condition: Optional[str] = ""
    current_symptoms: Optional[str] = ""
    food_allergies: Optional[str] = ""
    health_goals: List[str] = []

class NutritionGoals(BaseModel):
    macronutrients: Dict[str, float]
    vitamins: Dict[str, float]
    minerals: Dict[str, float]

class Food(BaseModel):
    id: str
    name: str
    category: str
    macronutrients: Dict
    vitamins: Optional[Dict] = {}
    minerals: Optional[Dict] = {}
    rasa: List[str] = []
    virya: Optional[str] = ""
    vipaka: Optional[str] = ""
    dosha_effects: Optional[Dict] = {}
    health_benefits: List[str] = []
    diet_type: str
    seasonal_availability: Optional[str] = ""

class DietGenerationRequest(BaseModel):
    patient_info: PatientInfo
    custom_nutrition_goals: Optional[NutritionGoals] = None
    available_foods: List[Food]
    requirements: Dict

class Query(BaseModel):
    question: str

# ============================================================================
# NEW ENDPOINT: /generate (for Diet Chart Auto-Fill)
# ============================================================================

@app.post("/generate")
async def generate_diet_chart(request: DietGenerationRequest):
    """
    Generate a complete 7-day diet chart with custom nutrition goals.
    This endpoint is called by your backend when doctor clicks 'AI Auto-Fill'.
    """
    try:
        patient = request.patient_info
        foods = request.available_foods
        custom_goals = request.custom_nutrition_goals
        
        if custom_goals is None:
            nutrition_goals = await generate_nutrition_goals(patient)
        else:
            nutrition_goals = custom_goals.dict()
        
        # Step 2: Build comprehensive prompt for diet generation
        diet_generation_prompt = build_diet_generation_prompt(
            patient=patient,
            foods=foods,
            nutrition_goals=nutrition_goals
        )
        
        # Step 3: Generate diet chart using your RAG chain
        diet_response = await asyncio.to_thread(
            chain.invoke, 
            diet_generation_prompt
        )
        
        # Step 4: Parse the AI response into structured format
        structured_meal_plan = await parse_diet_response_to_structure(
            diet_response, 
            foods
        )
        
        # Step 5: Return in the expected format
        return JSONResponse({
            "custom_nutrition_goals": nutrition_goals,
            "weekly_meal_plan": structured_meal_plan,
            "explanation": extract_explanation(diet_response),
            "considerations": extract_considerations(diet_response)
        })
        
    except Exception as e:
        print(f"Error generating diet chart: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def generate_nutrition_goals(patient: PatientInfo) -> Dict:
    """Generate personalized nutrition goals based on patient profile"""
    
    # Base goals by age and gender
    base_calories = 2000
    if patient.gender.lower() == "male":
        base_calories = 2200
    if patient.age > 50:
        base_calories -= 200
    if patient.age < 25:
        base_calories += 200
    
    # Adjust for constitution (Ayurvedic principle)
    constitution_adjustments = {
        "vata": {"calories": 0, "protein": 5, "fat": 10},
        "pitta": {"calories": 100, "protein": 10, "fat": -5},
        "kapha": {"calories": -200, "protein": 5, "fat": -10}
    }
    
    constitution = patient.constitution.lower()
    adjustments = constitution_adjustments.get(constitution, {"calories": 0, "protein": 0, "fat": 0})
    
    calories = base_calories + adjustments["calories"]
    protein = 50 + adjustments["protein"]
    fat = 65 + adjustments["fat"]
    carbs = int((calories - (protein * 4) - (fat * 9)) / 4)
    
    return {
        "macronutrients": {
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fat": fat,
            "fiber": 25
        },
        "vitamins": {
            "vitamin_a": 700 if patient.gender.lower() == "female" else 900,
            "vitamin_b1": 1.1,
            "vitamin_b2": 1.1,
            "vitamin_b3": 14,
            "vitamin_b6": 1.3,
            "vitamin_b12": 2.4,
            "vitamin_c": 75 if patient.gender.lower() == "female" else 90,
            "vitamin_d": 15,
            "vitamin_e": 15,
            "vitamin_k": 90,
            "folate": 400
        },
        "minerals": {
            "calcium": 1000,
            "iron": 18 if patient.gender.lower() == "female" else 10,
            "magnesium": 310 if patient.gender.lower() == "female" else 400,
            "phosphorus": 700,
            "potassium": 2600,
            "sodium": 1500,
            "zinc": 8 if patient.gender.lower() == "female" else 11
        }
    }


def build_diet_generation_prompt(patient: PatientInfo, foods: List[Food], nutrition_goals: Dict) -> str:
    """Build a comprehensive prompt for diet chart generation"""
    
    # Summarize available foods by category
    food_summary = {}
    for food in foods:
        category = food.category
        if category not in food_summary:
            food_summary[category] = []
        food_summary[category].append({
            "id": food.id,
            "name": food.name,
            "calories": food.macronutrients.get("calories_kcal", 0),
            "dosha": food.dosha_effects
        })
    
    # Build Ayurvedic recommendations
    constitution_guidance = get_constitution_guidance(patient.constitution)
    
    prompt = f"""You are AyurMind, an expert Ayurvedic Dietitian and Nutritionist.

PATIENT PROFILE:
- Name: {patient.name}
- Age: {patient.age} years, Gender: {patient.gender}
- Constitution (Prakriti): {patient.constitution}
- Health Condition: {patient.primary_health_condition or "None specified"}
- Current Symptoms: {patient.current_symptoms or "None"}
- Food Allergies: {patient.food_allergies or "None"}
- Health Goals: {", ".join(patient.health_goals) if patient.health_goals else "General wellness"}

NUTRITION TARGETS (Daily):
- Calories: {nutrition_goals['macronutrients']['calories']} kcal
- Protein: {nutrition_goals['macronutrients']['protein']}g
- Carbohydrates: {nutrition_goals['macronutrients']['carbs']}g
- Fat: {nutrition_goals['macronutrients']['fat']}g
- Fiber: {nutrition_goals['macronutrients']['fiber']}g

AYURVEDIC GUIDANCE FOR {patient.constitution.upper()}:
{constitution_guidance}

AVAILABLE FOODS BY CATEGORY:
{json.dumps(food_summary, indent=2)}

TASK:
Generate a COMPLETE 7-day personalized Ayurvedic diet chart with 4 meals per day (Breakfast, Lunch, Snacks, Dinner).

REQUIREMENTS:
1. Balance {patient.constitution} dosha using appropriate foods
2. Address: {patient.primary_health_condition or "general wellness"}
3. Avoid: {patient.food_allergies or "no restrictions"}
4. Meet daily nutrition targets (approximately)
5. Include variety across the week
6. Consider meal timing per Ayurveda (light breakfast, heavy lunch, moderate dinner)
7. Use foods from the available list ONLY

OUTPUT FORMAT:
For each day (Monday through Sunday) and each meal, provide:

**Day X:**
- **Breakfast:** [List 2-3 foods with amounts in grams]
- **Lunch:** [List 3-4 foods with amounts in grams]
- **Snacks:** [List 1-2 foods with amounts in grams]
- **Dinner:** [List 2-3 foods with amounts in grams]

Then provide:
- Brief explanation of why this diet works for this patient
- Key considerations or tips

EXAMPLE FORMAT:
**Monday:**
- **Breakfast:** Oats (50g), Almonds (20g), Banana (100g)
- **Lunch:** Brown Rice (150g), Dal (100g), Spinach (80g), Yogurt (50g)
- **Snacks:** Apple (100g), Walnuts (15g)
- **Dinner:** Roti (2 pieces, 100g), Mixed Vegetables (120g), Buttermilk (150ml)

Now generate the complete 7-day chart:"""
    
    return prompt


def get_constitution_guidance(constitution: str) -> str:
    """Get Ayurvedic dietary guidance based on constitution"""
    guidance = {
        "vata": """
- Favor: Warm, moist, grounding foods (cooked grains, root vegetables, ghee)
- Include: Sweet, sour, and salty tastes
- Avoid: Cold, raw, dry foods
- Best: Regular meal times, warm beverages
""",
        "pitta": """
- Favor: Cool, refreshing foods (cucumber, mint, coconut)
- Include: Sweet, bitter, and astringent tastes
- Avoid: Spicy, sour, salty foods
- Best: Moderate portions, avoid skipping meals
""",
        "kapha": """
- Favor: Light, dry, warm foods (barley, millet, legumes)
- Include: Pungent, bitter, and astringent tastes
- Avoid: Heavy, oily, cold foods
- Best: Lighter meals, warming spices
""",
        "vata-pitta": """
- Balance between warm (not hot) and cooling foods
- Emphasize sweet and bitter tastes
- Moderate portions with regular timing
""",
        "pitta-kapha": """
- Light, cooling foods with mild spices
- Emphasize bitter and astringent tastes
- Avoid heavy, oily foods
""",
        "vata-kapha": """
- Warm, light, easily digestible foods
- Emphasize pungent and bitter tastes
- Regular meal schedule
"""
    }
    return guidance.get(constitution.lower(), guidance["vata"])


async def parse_diet_response_to_structure(diet_response: str, foods: List[Food]) -> Dict:
    """
    Parse AI-generated diet chart text into structured JSON format.
    This is the critical function that converts AI text to the required format.
    """
    
    # Create food lookup by name (case-insensitive)
    food_lookup = {}
    for food in foods:
        food_lookup[food.name.lower()] = food.id
    
    weekly_plan = {
        "Mon": {"Breakfast": [], "Lunch": [], "Snacks": [], "Dinner": []},
        "Tue": {"Breakfast": [], "Lunch": [], "Snacks": [], "Dinner": []},
        "Wed": {"Breakfast": [], "Lunch": [], "Snacks": [], "Dinner": []},
        "Thu": {"Breakfast": [], "Lunch": [], "Snacks": [], "Dinner": []},
        "Fri": {"Breakfast": [], "Lunch": [], "Snacks": [], "Dinner": []},
        "Sat": {"Breakfast": [], "Lunch": [], "Snacks": [], "Dinner": []},
        "Sun": {"Breakfast": [], "Lunch": [], "Snacks": [], "Dinner": []}
    }
    
    # Parse the response
    lines = diet_response.split("\n")
    current_day = None
    current_meal = None
    
    day_mapping = {
        "monday": "Mon", "tuesday": "Tue", "wednesday": "Wed",
        "thursday": "Thu", "friday": "Fri", "saturday": "Sat", "sunday": "Sun",
        "day 1": "Mon", "day 2": "Tue", "day 3": "Wed",
        "day 4": "Thu", "day 5": "Fri", "day 6": "Sat", "day 7": "Sun"
    }
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Detect day
        for day_key, day_abbr in day_mapping.items():
            if day_key in line_lower and ("**" in line or "Day" in line):
                current_day = day_abbr
                break
        
        # Detect meal
        if current_day:
            if "breakfast" in line_lower:
                current_meal = "Breakfast"
            elif "lunch" in line_lower:
                current_meal = "Lunch"
            elif "snack" in line_lower:
                current_meal = "Snacks"
            elif "dinner" in line_lower:
                current_meal = "Dinner"
            
            # Extract foods from line
            if current_meal and (":" in line or "-" in line):
                foods_in_line = extract_foods_from_line(line, food_lookup)
                weekly_plan[current_day][current_meal].extend(foods_in_line)
    
    return weekly_plan


def extract_foods_from_line(line: str, food_lookup: Dict[str, str]) -> List[Dict]:
    """Extract individual food items with amounts from a line"""
    foods = []
    
    # Pattern: Food Name (amount unit)
    # Example: "Oats (50g), Almonds (20g)"
    pattern = r'([A-Za-z\s]+)\s*\((\d+)\s*([a-z]+)\)'
    matches = re.findall(pattern, line, re.IGNORECASE)
    
    for match in matches:
        food_name = match[0].strip().lower()
        amount = int(match[1])
        unit = match[2].lower()
        
        # Find matching food ID
        food_id = None
        for stored_name, stored_id in food_lookup.items():
            if food_name in stored_name or stored_name in food_name:
                food_id = stored_id
                break
        
        if food_id:
            foods.append({
                "food_id": food_id,
                "amount": amount,
                "serving_unit": unit
            })
    
    return foods


def extract_explanation(diet_response: str) -> str:
    """Extract explanation section from AI response"""
    explanation_markers = ["explanation:", "why this works:", "rationale:"]
    lines = diet_response.split("\n")
    
    explanation = []
    capturing = False
    
    for line in lines:
        line_lower = line.lower()
        if any(marker in line_lower for marker in explanation_markers):
            capturing = True
            continue
        if capturing and line.strip():
            if line.startswith("**") or "considerations" in line_lower:
                break
            explanation.append(line.strip())
    
    return " ".join(explanation[:3]) if explanation else "This personalized diet chart is designed based on Ayurvedic principles to balance your constitution and support your health goals."


def extract_considerations(diet_response: str) -> List[str]:
    """Extract key considerations from AI response"""
    considerations = []
    lines = diet_response.split("\n")
    
    capturing = False
    for line in lines:
        line_lower = line.lower()
        if "considerations" in line_lower or "tips" in line_lower or "notes" in line_lower:
            capturing = True
            continue
        if capturing and line.strip():
            if line.startswith("-") or line.startswith("•"):
                considerations.append(line.strip().lstrip("-•").strip())
            if len(considerations) >= 5:
                break
    
    return considerations if considerations else [
        "Eat at regular times",
        "Stay hydrated throughout the day",
        "Include warming spices in your meals"
    ]



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

app = FastAPI()
chain = None
retriever = None
rebuild_lock = threading.Lock()

origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
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

# --- MEMORY ---
conversation_histories = {}

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

