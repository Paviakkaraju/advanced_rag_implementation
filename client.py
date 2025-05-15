import requests
import json

def send_request_to_server(user_input):
    url = 'http://localhost:8000/tool/retrieve_chunks'
    headers = {'Content-Type': 'application/json'}

    payload = {"query": user_input}
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    try:
        return response.json()
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        return {"error": "Invalid response from server", "details": response.text}


