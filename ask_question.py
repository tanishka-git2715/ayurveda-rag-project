import requests
import json

url = "http://127.0.0.1:8000/ask"

my_question = "What are the three doshas in Ayurveda?"

payload = {"question": my_question}

headers = {"Content-Type": "application/json"}

print(f"Asking: {my_question}")

try:
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    response.raise_for_status() 

    print("\nReceived Answer:")
    print(response.json())

except requests.exceptions.RequestException as e:
    print(f"\nAn error occurred: {e}")
