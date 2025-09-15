import requests
import json

# The URL of your running FastAPI application
url = "https://tanishka2005-ayurveda-rag.hf.space/ask"

# The question you want to ask
my_question = "What are the three doshas in Ayurveda?"

# The data payload for the POST request
payload = {"question": my_question}

# Set the headers
headers = {"Content-Type": "application/json"}

print(f"Asking: {my_question}")

try:
    # Send the POST request
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    # Print the JSON response from the server
    print("\nReceived Answer:")
    print(response.json())

except requests.exceptions.RequestException as e:
    print(f"\nAn error occurred: {e}")