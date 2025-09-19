import os
import uvicorn  
from dotenv import load_dotenv
from pyngrok import ngrok

load_dotenv()

ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))

public_url = ngrok.connect(8000)
print(f"Public API available at: {public_url}")

uvicorn.run("app:app", host="127.0.0.1", port=8000)