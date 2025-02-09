from telegram.ext import ContextTypes

import requests
from typing import List, Tuple


async def generate_gpt_response(document_id, context: ContextTypes.DEFAULT_TYPE, url):
    URL = f"{url}/generate-roadmap/"
    user_request =  context.user_data.get('user_query')

    data = {"user_request":user_request,
            "document_id": document_id}

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(URL, json = data,headers=headers)

    if response.status_code==200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")



async def search_response(context: ContextTypes.DEFAULT_TYPE, url: str) -> List[Tuple[int, str]]:
    base_url = url
    search_endpoint = "/search"
    query_params = {
        "query": context.user_data["user_query"]
    }
    response = requests.get(base_url + search_endpoint, params=query_params)



    if response.status_code == 200:
        results = response.json().get('results', [])
        formatted_results = [
            (index, result['document_name'])
            for index, result in enumerate(results)
        ]
        document_ids = [result['document_id'] for result in results]
        context.user_data['search_results'] = formatted_results
        context.user_data['search_id'] = document_ids
        return formatted_results
    else:
        print(f"Ошибка при отправке запроса {response.status_code}: {response.text}")
        return []