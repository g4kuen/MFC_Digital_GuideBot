import json
import asyncio
import aiohttp


async def fetch_api_data(api_url, data, session=None):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=data) as response:
                response.raise_for_status()
                response_json = await response.json()
                return response_json["info"]["content"]
    except aiohttp.ClientResponseError as e:
        print(f"An HTTP error occurred: {e}")
        print(f"Data with error {data}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_links_by_doc(docs):
    doc_links_dict = {}
    for sub_doc in docs:
        doc_links = sub_doc["links"]
        for link in doc_links:
            doc_name = link["name"]
            doc_links_dict[doc_name] = link["link_docid"]

    return doc_links_dict


async def filter_data_by_section(api_url, data):
    # links = []
    links = {}  # for themes
    error_docs = []

    async with aiohttp.ClientSession() as session:
        # Gather tasks for all documents
        tasks = []
        for document in data:
            try:
                section = document["params"]["integrationcode"]
                data_to_send = {
                    "token": "156f7191-7701-4d05-9ba6-5afcf6b701c7",
                    "method": "MFC_CV_GetContent",
                    "data": {
                        "id": "3613",
                        "section": section
                    }
                }
                # Create a task for each document
                task = fetch_api_data(api_url, data_to_send, session)
                tasks.append((document, task))
            except KeyError:
                error_docs.append(document)

        # Await all tasks and process results
        for document, task in tasks:
            try:
                docs_associated_with_section = await task
                if docs_associated_with_section:
                    doc_links_dict = get_links_by_doc(docs_associated_with_section)
                    # links.append(doc_links_dict)
                    document_name = document["params"]["displayname"]
                    links[document_name] = doc_links_dict
            except Exception as e:
                print(f"Error processing document {document}: {e}")
                error_docs.append(document)

    # Write results to JSON files
    with open('doc_links_with_theme.json', 'w', encoding="utf-8") as f:
        json.dump(links, f, indent=4)

    with open('error_docs_situation.json', 'w') as f:
        json.dump(error_docs, f, indent=4)


async def main_async(api_url, initial_data):
    async with aiohttp.ClientSession() as session:
        fetched_data = await fetch_api_data(api_url, initial_data, session)

    await filter_data_by_section(api_url, fetched_data)


def main(api_url, initial_data):
    asyncio.run(main_async(api_url, initial_data))


api_endpoint = "https://forcase.mfcto.ru/api/"
data_to_send = {
    "token": "...",
    "method": "MFC_CV_GetContent",
    "data": {
        "id": "3613",
        "section": "uslugi_v_razreze_organov_vlasti"
    }
}

main(api_endpoint, data_to_send)