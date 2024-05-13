import requests
import json
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Define the URL and headers
url = "http://127.0.0.1:5433/app/otter"
headers = {
    "Content-Type": "application/json"
}

data_payload = {
    "content": [
        {
            "prompt": "Inventory: None\nTask Goal: installing_a_fax_machine\nOriginal Subtasks: None\nPrevious Action Code: No code\nExecution error: No error\nNow, please output Explain, Subtasks (revise if necessary), Code that completing the next subtask, and Target States, according to the instruction above. Remember you can only use the functions provided above and pay attention to the response format.",
            "images": {
                'image0': image_to_base64('./data/rgb0_detect_surroundings.png'),
                'image1': image_to_base64('./data/rgb1_detect_surroundings.png'),
                'image2': image_to_base64('./data/rgb2_detect_surroundings.png'),
                'image3': image_to_base64('./data/rgb3_detect_surroundings.png'),
                }
        }
    ],
    "token": "sk-ICLRBESTPAPERAWARDOCTOPUS"
}

# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(data_payload))
print(response.text)
