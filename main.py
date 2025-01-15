import json
import re

import nltk
import numpy as np
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bot import get_document_vector


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
with open('data/doc_links_situation_stemming.json', 'r', encoding='utf-8') as file:
    raw_label_data = json.load(file)
documents = []
for item in raw_label_data:
    for key, value in item.items():
        documents.append(f"{key} ` {value}")

preprocessed_documents = [preprocess_text(doc) for doc in documents]


vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(preprocessed_documents)

def find_top_similar_documents(text, top_n=3):

    query_vector = vectorizer.transform([text])


    similarities = cosine_similarity(query_vector, document_vectors)[0]


    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [(index, documents[index], similarities[index]) for index in top_indices]


def find_threshold_similar_documents(text, threshold=0.1):

    # Получаем вектор для запроса
    query_vector = vectorizer.transform([text])

    # Вычисляем сходство между запросом и всеми документами
    similarities = cosine_similarity(query_vector, document_vectors)[0]

    # Отбираем документы, у которых сходство больше или равно порогу
    top_results = [
        (index, documents[index], similarity)
        for index, similarity in enumerate(similarities)
        if similarity >= threshold
    ]

    return top_results

print("Добро пожаловать! Введите запрос, чтобы найти самые похожие документы. Введите 'exit' для выхода.")
while True:
    user_input = input("Введите ваш запрос: ")
    if user_input.lower() == 'exit':
        print("Выход из программы. До свидания!")
        break

    user_input=stem_text(user_input)
    top_results = find_top_similar_documents(user_input, top_n=5)
    print("\nТоп-5 похожих документа:")
    for rank, (index, text, similarity) in enumerate(top_results, start=1):
        print(f"{rank}. Документ {index} (Сходство: {similarity:.4f}): {text}\n")
    print(top_results)



#from gensim.models import Word2Vec
#from sklearn.metrics import euclidean_distances

# # cbow_model = Word2Vec(sentences=tokenized_documents, vector_size=100, window=9, min_count=1, sg=1)
# #
# #
# # document_vectors = np.array([get_document_vector(doc, cbow_model) for doc in tokenized_documents])
# #
# # user_message="льготный проезд"
# #
# # user_tokens = preprocess_text(user_message)
# # print(user_tokens)
# # user_vector = get_document_vector(user_tokens, cbow_model)
# #
# # euclidean_distances_results = euclidean_distances([user_vector], document_vectors).flatten()
#
# top_indices = euclidean_distances_results.argsort()[-5:]
# print(top_indices)
# print(euclidean_distances_results[top_indices])
