import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

# --- CONFIGURATION ---
load_dotenv()
DATA_PATH = "data/"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PINECONE_INDEX_NAME = "ayurveda-rag"

def populate_vectorstore():
    """
    Reads documents, creates embeddings, and uploads them to Pinecone.
    """
    print("Starting data ingestion process...")

    # --- 1. Load Documents ---
    documents = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            path = os.path.join(root, file)
            print(f"--> Processing file: {file}")
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    documents.extend(loader.load())
                elif file.endswith(".txt"):
                    loader = TextLoader(path, encoding="utf-8")
                    documents.extend(loader.load())
            except Exception as e:
                print(f"!! ERROR skipping {file}: {e}")

    if not documents:
        print("No documents found. Aborting.")
        return
    print(f"Loaded {len(documents)} documents.")

    # --- 2. Split Documents into Chunks ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # --- 3. Initialize Embeddings Model ---
    print("Initializing Google Embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY
    )

    # --- 4. Initialize Pinecone ---
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = pc.Index(PINECONE_INDEX_NAME)

    # --- 5. Upsert Chunks to Pinecone ---
    print("Upserting chunks to Pinecone... This may take a while.")
    # Process in batches to avoid overwhelming the API
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk.page_content for chunk in batch]
        
        # Create embeddings for the batch
        embedded_texts = embeddings.embed_documents(texts)
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        for j, chunk in enumerate(batch):
            vector = {
                "id": f"chunk_{i+j}",
                "values": embedded_texts[j],
                "metadata": {"text": chunk.page_content}
            }
            vectors_to_upsert.append(vector)

        # Upsert the batch
        index.upsert(vectors=vectors_to_upsert)
        print(f"Upserted batch {i//batch_size + 1}")

    print("\n--- Data ingestion complete! Your Pinecone index is ready. ---")

if __name__ == "__main__":
    populate_vectorstore()