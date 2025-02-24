import requests
import json

# API Endpoint (Replace with the actual working API URL)
API_URL = "http://serv-image-scan.veeka.private/green/image/scan"
# Image URL with words (Replace with the actual image URL you want to scan)
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Lorem_Ipsum_1.png/800px-Lorem_Ipsum_1.png"

# Request Payload
payload = {
    "bizType": "default",
    "scenes": ["ocr"],  # Ensures OCR processing
    "source": "test",
    "tasks": [
        {
            "url": IMAGE_URL,  # Image to be scanned
            "dataId": "yy",
            "taskId": "xx",
            "moduleTable": ""
        }
    ]
}

# Headers
headers = {
    "Content-Type": "application/json"
}

# Send POST request to the API
response = requests.post(API_URL, data=json.dumps(payload), headers=headers)

# Check response status
if response.status_code == 200:
    result = response.json()
    
    # Extract OCR data
    detected_texts = []
    
    for item in result.get("data", []):
        for res in item.get("results", []):
            if "ocrData" in res:
                detected_texts.extend(res["ocrData"])
    
    # Print extracted text
    print("Extracted OCR Words:")
    for word in detected_texts:
        print("-", word)

else:
    print("Error calling API:", response.status_code, response.text)
