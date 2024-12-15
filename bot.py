import os

import json
import nltk
import re
from dotenv import load_dotenv
from fontTools.merge.util import first
from numpy.random import choice
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from transformers import pipeline



import requests





load_dotenv()
TOKEN = os.getenv("TOKEN")
url = os.getenv("URL")
with open('doc_links_situation1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

with open('doc_links_situation1.json', 'r', encoding='utf-8') as file:
    raw_label_data = json.load(file)

with open('document_text_situation.json','r',encoding='utf-8') as file:
    docs_text = json.load(file)

label_mapping = {}
for obj in raw_label_data:
    for key, value in obj.items():
        label_mapping[str(value)] = key

stemmer = SnowballStemmer("russian")
stop_words = set(stopwords.words('russian'))


def preprocess_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\[\]]', '', text)
    text = text.lower()

    return text



def generate_gpt_response(service_name):
    last_number = None
    numbers = re.findall(r'\d+', service_name)
    if numbers:
        last_number = numbers[-1]

    id = last_number

    # if id in docs_text:
    #     document_text = docs_text[id]
    #
    #     if isinstance(document_text, str):
    #         words = document_text.split()
    #         first_1000_words = words[:300]
    #         first_1000_words_text = ' '.join(first_1000_words)
    #         document_text = first_1000_words_text
    #         print(type(document_text))
    #     else:
    #         print("Ошибка: document_text не является строкой. Преобразуем в строку.")
    #         document_text = str(document_text)
    #         words = document_text.split()
    #         first_1000_words = words[:300]
    #         first_1000_words_text = ' '.join(first_1000_words)
    #         document_text = first_1000_words_text
    #         print(type(document_text))
    # document_text = preprocess_text(document_text)
    # print("doc_text " ,document_text)
    # print("len text ", len(document_text))

    prompt = f"Документ ID {last_number}\n\nОпиши, какие шаги нужно предпринять для оформления услуги: {service_name}"
    #prompt = f"Документ ID {last_number}\n\nОпиши, какие шаги нужно предпринять для оформления услуги: {service_name} текст для изучения :{document_text}. текст результата:"
    print(prompt)

    data = {"text": prompt}

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Ответ нейронной сети:")
        print(response.json()['response'])
    else:
        print("Ошибка при отправке запроса:", response.status_code)

    generated_text = response.json()['response']
    generated_text = generated_text[len(prompt):]
    print(generated_text)
    return generated_text


documents = []
for item in raw_label_data:
    for key, value in item.items():
        documents.append(f"{key} {value}")

tokenized_documents = [preprocess_text(doc) for doc in documents]

cbow_model = Word2Vec(sentences=tokenized_documents, vector_size=100, window=5, min_count=1, sg=0)


def document_vector(doc, model):
    doc = [word for word in doc if word in model.wv.index_to_key]
    if doc:
        return np.mean(model.wv[doc], axis=0)
    else:
        return np.zeros(model.vector_size)


document_vectors = np.array([document_vector(doc, cbow_model) for doc in tokenized_documents])


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


def split_message(message, max_length=4096):
    return [message[i:i + max_length] for i in range(0, len(message), max_length)]



def generate_choice_keyboard(indices):
    keyboard = [
        [InlineKeyboardButton(f"Ответ {i + 1}", callback_data=f"choose_{i + 1}")]
        for i in range(len(indices))
    ]
    return InlineKeyboardMarkup(keyboard)



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Составить запрос", callback_data='compose_request')],
        [InlineKeyboardButton("Оставить отзыв", callback_data='leave_feedback')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите действие:", reply_markup=reply_markup)



async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == 'compose_request':
        await query.edit_message_text(text="Составьте запрос.")
    elif query.data == 'leave_feedback':
        await query.edit_message_text(text="Составьте отзыв.")
        context.user_data['waiting_for_feedback'] = True

top_indixes=[]

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get('waiting_for_feedback'):
        feedback = update.message.text
        print("составленный отзыв ", feedback)
        await update.message.reply_text(f"Вы составили отзыв: {feedback}")
        context.user_data['waiting_for_feedback'] = False
    else:
        user_message = update.message.text
        if 'user_query' in context.user_data:
            user_message = context.user_data['user_query'] + " " + user_message
        user_tokens = preprocess_text(user_message)
        user_vector = document_vector(user_tokens, cbow_model)
        cosine_similarities = cosine_similarity([user_vector], document_vectors).flatten()
        top_indices = cosine_similarities.argsort()[-5:][::-1]

        context.user_data['top_indices'] = top_indices
        top_indixes = context.user_data['top_indices']

        if cosine_similarities[top_indices[0]] < 0.4:
            if 'query_attempts' not in context.user_data:
                context.user_data['query_attempts'] = 1
                context.user_data['user_query'] = user_message
                await update.message.reply_text(f"Ваш текущий запрос: {user_message}\nМы не нашли похожих тем, пожалуйста, дополните запрос.")
            elif context.user_data['query_attempts'] < 3:
                context.user_data['query_attempts'] += 1
                updated_query = user_message
                context.user_data['user_query'] = updated_query
                await update.message.reply_text(f"Ваш текущий запрос: {updated_query}\nМы не нашли похожих тем, пожалуйста, дополните запрос.")
            else:
                context.user_data['query_attempts'] = 0
                context.user_data['user_query'] = ""
                await update.message.reply_text("Ваш запрос не дал нужных результатов. Пожалуйста, перепишите запрос.")
                return
        else:
            context.user_data['user_query'] = ""
            response = get_labeled_response(top_indices)
            message_parts = split_message(f"Найденные похожие записи:\n\n{response}")
            for part in message_parts:
                await update.message.reply_text(part)

            choice_keyboard = generate_choice_keyboard(top_indices)

            await update.message.reply_text("Выберите один из ответов:", reply_markup=choice_keyboard)

async def choice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    print(context.user_data)
    top_indices = context.user_data['top_indices']
    if top_indices.size == 0:
        await query.edit_message_text("Ошибка: нет доступных индексов для выбора.")
        return

    choice_number = int(query.data.split('_')[1]) - 1
    selected_service = documents[top_indices[choice_number]].split(" ", 1)[-1]
    selected_index=top_indices[choice_number]


    gpt_response = generate_gpt_response(selected_service)

    await query.edit_message_text(f"Вы выбрали услугу: {selected_service}\n\nГенерация: {gpt_response}")


def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_handler, pattern='^(compose_request|leave_feedback)$'))
    application.add_handler(CallbackQueryHandler(choice_handler, pattern=r'^choose_\d+$'))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("started")
    application.run_polling()


if __name__ == '__main__':
    main()
