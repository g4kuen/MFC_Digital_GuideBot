import os

import requests
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("URL")

data = {"text": "напиши сортировку пузырьком на python"}

# response = requests.post(url, json=data)
#
# if response.status_code == 200:
#     print("Ответ нейронной сети:")
#     print(response.json()['response'])
# else:
#     print("Ошибка при отправке запроса:", response.status_code)

def test_search():
    base_url = url
    search_endpoint="/search"
    query_params = {
        "query": "льготный проезд"
    }
    response = requests.get(base_url+search_endpoint,params=query_params)

    if response.status_code == 200:
        print("Ответ нейронной сети:")
        print(response.json()['query'])
        print(response.json()['results'])
    else:
        print(f"Ошибка при отправке запроса {response.status_code}: {response.text}")
test_search()