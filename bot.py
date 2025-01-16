import os

import json
from types import NoneType

import nltk
import re
from dotenv import load_dotenv
from fontTools.merge.util import first
from gensim.parsing import stem_text
from numpy.random import choice
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes, \
    CallbackContext
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from torch.ao.nn.quantized.functional import threshold
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import requests


load_dotenv()
TOKEN = os.getenv("TOKEN")
#url = os.getenv("URL")

with open('data/doc_links_situation_stemming.json', 'r', encoding='utf-8') as file:
    raw_label_data = json.load(file)

with open('data/doc_links1.json', 'r', encoding='utf-8') as file:
     data = json.load(file)

with open('data/document_text_situation.json','r',encoding='utf-8') as file:
    docs_text = json.load(file)



documents = []
for item in raw_label_data:
    for key, value in item.items():
        documents.append(f"{key} ` {value}")



label_mapping = {}
for obj in raw_label_data:
    for key, value in obj.items():
        label_mapping[str(value)] = key

stemmer = SnowballStemmer("russian")
stop_words = set(stopwords.words('russian'))

def stem_text(text):#stem ru
    stemmer = SnowballStemmer("russian")
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def preprocess_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s:]', '', text)
    text = re.sub(r'[\[\]]', '', text)
    text = text.lower()

    return text


preprocessed_documents = [preprocess_text(doc) for doc in documents]
vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(preprocessed_documents)



# def find_top_similar_documents(text, top_n=3):
#
#     query_vector = vectorizer.transform([text])
#
#     similarities = cosine_similarity(query_vector, document_vectors)[0]
#
#     threshold_indices = np.argsort(similarities)[-top_n:][::-1]
#
#     return [(index, documents[index], similarities[index]) for index in threshold_indices]



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Составить запрос ", callback_data='compose_request')],
        [InlineKeyboardButton("Оставить отзыв ", callback_data='leave_feedback')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите действие:", reply_markup=reply_markup)



async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data['waiting_for_query'] = False
    await update.message.reply_text("Вы остановили прием запросов, для запуска выберите 'Составить запрос' в команде /start")



def split_message(message, max_length=4096):
    return [message[i:i + max_length] for i in range(0, len(message), max_length)]


def find_threshold_similar_documents(text, threshold=0.1):
    query_vector = vectorizer.transform([text])

    similarities = cosine_similarity(query_vector, document_vectors)[0]

    top_results = [
        (index, documents[index], similarity)
        for index, similarity in enumerate(similarities)
        if similarity >= threshold
    ]

    return top_results



def generate_gpt_response(service_name):
    last_number = None
    numbers = re.findall(r'\d+', service_name)
    if numbers:
        last_number = numbers[-1]

    prompt = f"Документ ID {last_number}\n\nОпиши, какие шаги нужно предпринять для оформления услуги: {service_name.split('`')[0]}"
    #prompt = f"Документ ID {last_number}\n\nОпиши, какие шаги нужно предпринять для оформления услуги: {service_name} текст для изучения :{document_text}. текст результата:"
    #print(prompt)

    #   data = {"text": prompt}
    #response = requests.post(url, json=data)

    # generated_text = response.json()['response']
    # generated_text = generated_text[len(prompt):]

    generated_text = prompt # пока ответов не будет
    return generated_text



def get_document_vector(doc, model):
    doc = [word for word in doc if word in model.wv.index_to_key]
    if doc:
        return np.mean(model.wv[doc], axis=0)
    else:
        return np.zeros(model.vector_size)



def get_labeled_response(indices):
    labeled_responses = []
    for idx, index in enumerate(indices, start=1):
        original_document = documents[index].split(" ", 1)[-1]
        doc_id = re.search(r'\b\d+\b', documents[index])
        if doc_id:
            doc_id = doc_id.group(0)
            label = label_mapping.get(doc_id, "Метка не найдена")
        else:
            label = "Метка не найдена"
        labeled_responses.append(f"{idx}. {label}")
    return "\n\n".join(labeled_responses)



def generate_choice_keyboard(indices,flag):

    keyboard=[]
    if flag == True:
        keyboard = [
            [InlineKeyboardButton("⬅️ Назад", callback_data="prev_page"),
             InlineKeyboardButton("Вперёд ➡️", callback_data="next_page")]
        ]

        keyboard += [
            [InlineKeyboardButton(f"Ответ {i + 1}", callback_data=f"choose_{i + 1}")]
            for i in range(len(indices))
        ]
    else:
        keyboard = [[ InlineKeyboardButton(f"Ответ {i + 1}", callback_data=f"choose_{i + 1}")]
            for i in range(len(indices))
        ]




    return InlineKeyboardMarkup(keyboard)



async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    print(query)
    await query.answer()



async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == 'compose_request':
        await query.edit_message_text(text="Составьте запрос. Чтобы остановить прием запросов, выполните команду /stop")
        context.user_data['waiting_for_query'] = True
    if query.data == 'leave_feedback':
        await query.edit_message_text(text="Составьте отзыв.")
        context.user_data['waiting_for_feedback'] = True
    if query.data == 'refine_query':
        context.user_data['refine_mode'] = True
        await query.edit_message_text(text=f"Пожалуйста, уточните ваш запрос, добавив дополнительную информацию. текущий запрос : {context.user_data['user_query']} ")
    elif query.data == 'generate_new_query':
        context.user_data['refine_mode'] = False
        context.user_data['query_attempts'] = 0
        context.user_data['user_query'] = ""
        await query.edit_message_text(text="Пожалуйста, введите новый запрос.")


def create_query_buttons():
    keyboard = [
        [InlineKeyboardButton("Доуточнить запрос", callback_data="refine_query")],
        [InlineKeyboardButton("Сгенерировать новый запрос", callback_data="generate_new_query")]
    ]
    return InlineKeyboardMarkup(keyboard)



async def handle_empty_results(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query_attempts = context.user_data.get('query_attempts')
    refine_mode = context.user_data.get('refine_mode')
    current_query = context.user_data.get('user_query', '')

    if refine_mode:

        new_query_part = update.message.text
        updated_query = f"{current_query}{new_query_part}".strip()
        context.user_data['user_query'] = updated_query
        context.user_data['query_attempts'] = query_attempts + 1
        await update.message.reply_text(
            f"Ваш текущий уточненный запрос: {updated_query}"
        )

    else:
        current_query = update.message.text
        context.user_data['user_query']=current_query


    if query_attempts < 2:
        print(context.user_data['query_attempts'])


        if not refine_mode:
            await update.message.reply_text(
                f"Мы не нашли похожих тем. Вы можете уточнить свой запрос или попробовать создать новый, ваш текущий запрос: {current_query}",
                reply_markup=create_query_buttons()
            )

    else:
        context.user_data['query_attempts'] = 0
        context.user_data['user_query'] = ""
        context.user_data['refine_mode'] = False
        await update.message.reply_text("Ваш запрос не дал результатов. Пожалуйста, перепишите запрос.")



async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get('waiting_for_feedback'):
        feedback = update.message.text
        ###обработка для feedback
        await update.message.reply_text(f"Вы составили отзыв: {feedback}")
        context.user_data['waiting_for_feedback'] = False

    if context.user_data.get('waiting_for_query'):
        user_message = update.message.text
        if 'query_attempts' not in context.user_data:
            context.user_data['query_attempts'] = 1
        if 'user_query' in context.user_data:
            user_message = context.user_data['user_query'] + " " + user_message
        if 'user_query' not in context.user_data or context.user_data['user_query'] == "":
            context.user_data['user_query']=user_message

        flag = False
        user_input = preprocess_text(user_message)
        preprocessed_text = preprocess_text(user_input)
        stemmed_text = stem_text(preprocessed_text)
        threshold_results = find_threshold_similar_documents(stemmed_text, 0.1)

        if len(threshold_results) == 0:
            await handle_empty_results(update, context)
        else:
            context.user_data['user_query'] = ""
            context.user_data['query_attempts'] = 0

            # Сохранить результаты и индексы в user_data
            context.user_data['threshold_results'] = threshold_results
            context.user_data['current_page'] = 0


            page_results = get_page_results(threshold_results, context.user_data['current_page'])
            page_indices = [result[0] for result in page_results]  # Индексы для текущей страницы
            context.user_data['threshold_indices'] = page_indices


            if (len(threshold_results) > 5):
                flag = True
            else:
                flag = False

            choice_keyboard = generate_choice_keyboard(page_indices, flag)
            context.user_data['refine_mode'] = False

            response = "\n\n".join([f"{result[1].split('`')[0]}" for result in page_results])


            await update.message.reply_text(
                f"Найденные похожие записи:\n\n{response}",
                reply_markup=choice_keyboard
            )



async def button_query_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()

    max_page = len(context.user_data['threshold_results']) // 5

    if query.data == "prev_page":
        if context.user_data['current_page'] > 0:
            context.user_data['current_page'] -= 1
        else:
            context.user_data['current_page'] = max_page

    elif query.data == "next_page":
        if context.user_data['current_page'] < max_page:
            context.user_data['current_page'] += 1
        else:
            context.user_data['current_page'] = 0

    current_page = context.user_data['current_page']
    threshold_results = context.user_data['threshold_results']
    page_results = threshold_results[current_page * 5:(current_page + 1) * 5]
    page_indices = [result[0] for result in page_results]
    context.user_data['threshold_indices'] = page_indices

    response = "\n\n".join([f"{result[1].split('`')[0]}" for result in page_results])


    choice_keyboard=generate_choice_keyboard(page_indices, True)

    # if len(page_results) <5:
    #     await query.edit_message_reply_markup()
    #     await update.message.reply_text("Выберите один из ответов:", reply_markup=generate_choice_keyboard(page_results))

    await query.edit_message_text(
        text=f"Найденные похожие записи:\n\n{response}",
        reply_markup=choice_keyboard if choice_keyboard else None
    )



async def choice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()


    threshold_indices = context.user_data['threshold_indices']
    if len(threshold_indices) == 0:
        await query.edit_message_text("Ошибка: нет доступных индексов для выбора.")
        return

    choice_number = int(query.data.split('_')[1]) - 1
    selected_service = documents[threshold_indices[choice_number]]#.split(" ", 1)[-1]
    #selected_index = threshold_indices[choice_number]


    gpt_response = generate_gpt_response(selected_service)

    await query.edit_message_text(f"Вы выбрали услугу: {selected_service.split('`')[0]}\n\nГенерация: {gpt_response.split(':')[1]}")



def get_page_results(results, page, page_size=5):
    start = page * page_size
    end = start + page_size
    return results[start:end]



def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CallbackQueryHandler(button_handler, pattern='^(compose_request|leave_feedback|refine_query|generate_new_query)$'))
    application.add_handler(CallbackQueryHandler(button_query_handler,pattern='^(next_page|prev_page)$'))
    application.add_handler(CallbackQueryHandler(choice_handler, pattern=r'^choose_\d+$'))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("started")
    application.run_polling()


if __name__ == '__main__':
    main()
