import uvicorn
from pyngrok import ngrok
import os
from dotenv import load_dotenv

# Load .env so NGROK_AUTH_TOKEN is available
load_dotenv()

# Start ngrok tunnel
ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))
public_url = ngrok.connect(8000)
print(f"🚀 Public API available at: {public_url}")

# Run uvicorn with reload enabled
uvicorn.run("app:app", host="127.0.0.1", port=8000)
