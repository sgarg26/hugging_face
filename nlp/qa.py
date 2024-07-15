import requests
import pandas as pd


API_URL = "https://api-inference.huggingface.co/models/deepset/tinyroberta-squad2"
headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

df = pd.read_csv("players.csv")


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


q = input("Question: ")
inp = df.to_dict("split")["data"]

print(str(inp))

output = query(
    {
        "inputs": {
            "question": q,
            "context": str(inp),
        },
    }
)

print(output)
