import os

import requests
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("URL")

data = {"text": "напиши сортировку пузырьком на python"}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Ответ нейронной сети:")
    print(response.json()['response'])
else:
    print("Ошибка при отправке запроса:", response.status_code)
