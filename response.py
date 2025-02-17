from telegram.ext import ContextTypes

import aiohttp
from typing import List, Tuple
from telegram.ext import ContextTypes


async def generate_gpt_response(document_id, context: ContextTypes.DEFAULT_TYPE, url):
    URL = f"{url}/generate-roadmap/"
    user_request = context.user_data.get("user_query")

    data = {
        "user_request": user_request,
        "document_id": document_id
    }

    headers = {
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(URL, json=data, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Request failed with status code {response.status}: {await response.text()}")






async def search_response(context: ContextTypes.DEFAULT_TYPE, url: str) -> List[Tuple[int, str]]:
    base_url = url
    search_endpoint = "/search"
    query_params = {
        "query": context.user_data["user_query"]
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(base_url + search_endpoint, params=query_params) as response:
            if response.status == 200:
                data = await response.json()
                results = data.get('results', [])
                formatted_results = [
                    (index, result['document_name'])
                    for index, result in enumerate(results)
                ]
                document_ids = [result['document_id'] for result in results]


                context.user_data['search_results'] = formatted_results
                context.user_data['search_id'] = document_ids

                return formatted_results
            else:
                print(f"Ошибка при отправке запроса {response.status}: {await response.text()}")
                return []
