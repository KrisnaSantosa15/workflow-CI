import requests
import json

# Correct URL for the running MLflow model server
URL = "http://localhost:5001/invocations"

# Load input data
with open("model_dir/model/serving_input_example.json") as f:
    data = json.load(f)

# Make request
response = requests.post(URL, json=data)

# Show prediction
print("Prediction response:", response.text)
