import asyncio
import random

from telegram.ext import ContextTypes

import aiohttp
from typing import List, Tuple
from telegram.ext import ContextTypes

from MFC_Digital_GuideBot.logger import logger


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


async def fake_generate_gpt_response(document_id, context: ContextTypes.DEFAULT_TYPE, url):
    delay = random.uniform(10.0, 40.0)
    await asyncio.sleep(delay)

    return {
        'roadmap': "## Ваш ответ готов\n\n"
                   "1. Пример первого шага\n"
                   "2. Пример второго шага\n"
                   "3. Финальное действие\n\n"
                   "*Это тестовый ответ, сгенерированный заглушкой*",
        'document_id': document_id,
        'processing_time': delay
    }





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
                logger.info(f"Ошибка при отправке запроса {response.status}: {await response.text()}")
                #print(f"Ошибка при отправке запроса {response.status}: {await response.text()}")
                return []


async def fake_search_response(context: ContextTypes.DEFAULT_TYPE, url: str) -> List[Tuple[int, str]]:
    await asyncio.sleep(random.uniform(1.0, 3.0))

    fake_services = [
        "Получение справки о несудимости",
        "Оформление загранпаспорта",
        "Регистрация по месту жительства",
        "Постановка на налоговый учёт",
        "Оформление ИП",
    ]

    num_results = 5
    results = random.sample(fake_services, num_results)

    formatted_results = [(i, results[i]) for i in range(num_results)]
    document_ids = [f"doc_{random.randint(1000, 9999)}" for _ in range(num_results)]

    context.user_data['search_results'] = formatted_results
    context.user_data['search_id'] = document_ids

    return formatted_results
