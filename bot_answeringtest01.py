import os
from dotenv import load_dotenv
from gensim import corpora, models
from gensim.similarities import MatrixSimilarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim.utils
import re

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = gensim.utils.simple_preprocess(text)
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

with open('ap/ap.txt', 'r', encoding='utf-8') as file:
    documents = file.readlines()

texts = [preprocess_text(doc) for doc in documents]

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

num_topics = 100
model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    passes=10,
    alpha='auto',
    eta='auto'
)

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()
TOKEN = os.getenv("TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Напишите запрос, и я найду для вас 5 подходящих тем!")

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text

    # Предобработка запроса
    processed_query = preprocess_text(user_query)
    query_bow = dictionary.doc2bow(processed_query)

    # Определение распределения тем для запроса
    query_topics = model[query_bow]  # Это распределение по темам для запроса

    # Вычисляем сходство между запросом и всеми темами
    similarities = []
    for topic_id in range(num_topics):
        # Ищем вероятность для текущей темы в запросе
        topic_probability = next((prob for t_id, prob in query_topics if t_id == topic_id), 0)
        similarities.append((topic_id, topic_probability))

    # Сортировка по сходству и извлечение топ-5 тем
    top_topics = sorted(similarities, key=lambda item: -item[1])[:5]

    # Формирование ответа
    response = "Топ-5 подходящих тем:\n"
    for topic_id, similarity in top_topics:
        print(f'topicID: {topic_id}, similarity: {similarity}')
        print(f'numTopics: {num_topics}')

        if 0 <= topic_id < num_topics:
            # Извлекаем ключевые слова для темы
            keywords = model.show_topic(topic_id, topn=5)
            keywords_str = ", ".join([word for word, _ in keywords])
            response += f"Тема {topic_id + 1}: {keywords_str} (сходство: {similarity:.2f})\n"

    await update.message.reply_text(response)


def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    print("started")
    application.run_polling()

if __name__ == '__main__':
    main()
