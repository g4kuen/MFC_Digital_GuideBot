import json
import re
import nltk
from collections import defaultdict
from nltk.stem import PorterStemmer
import pymorphy2
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
# nltk.download('stopwords')
# nltk.download('punkt')
import json
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


with open('document_text.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('russian'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def stem_text(text):#stem ru
    stemmer = SnowballStemmer("russian")
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# def stem_text(text):
#     stemmer = PorterStemmer()
#     words = nltk.word_tokenize(text)
#     stemmed_words = [stemmer.stem(word) for word in words]
#     return ' '.join(stemmed_words)

def lemmatize_text(text):
    morph = pymorphy2.MorphAnalyzer()
    words = nltk.word_tokenize(text)
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmatized_words)


# preprocessed_data_stemming = {}
# preprocessed_data_lemmatize = {}
#
# for key, values in data.items():
#     print(key)
#     cleaned_values = [clean_text(value) for value in values]
#     cleaned_values = [remove_stopwords(value) for value in cleaned_values]
#
#     stemmed_values = [stem_text_russian(value) for value in cleaned_values]
#     lemmatized_values = [lemmatize_text(value) for value in cleaned_values]
#
#     preprocessed_data_stemming[key] = stemmed_values
#     preprocessed_data_lemmatize[key] = lemmatized_values
# with open('document_text_preprocessed_stemming.json', 'w', encoding='utf-8') as file:
#     json.dump(preprocessed_data_stemming, file, indent=4, ensure_ascii=False)
#
# with open('document_text_preprocessed_lemmatize.json', 'w', encoding='utf-8') as file:
#     json.dump(preprocessed_data_lemmatize, file, indent=4, ensure_ascii=False)

# preprocessed_data_stemming = []
# preprocessed_data_lemmatize = []
#
# for item in data:
#     cleaned_item = {}
#     for key, value in item.items():
#         cleaned_key = clean_text(key)
#         cleaned_key = remove_stopwords(cleaned_key)
#         cleaned_item[cleaned_key] = value
#
#     preprocessed_data_stemming.append({stem_text_russian(key): value for key, value in cleaned_item.items()})
#     preprocessed_data_lemmatize.append({lemmatize_text(key): value for key, value in cleaned_item.items()})
#
# with open('document_text_preprocessed_stemming.json', 'w', encoding='utf-8') as file:
#     json.dump(preprocessed_data_stemming, file, indent=4, ensure_ascii=False)
#
# with open('document_text_preprocessed_lemmatize.json', 'w', encoding='utf-8') as file:
#     json.dump(preprocessed_data_lemmatize, file, indent=4, ensure_ascii=False)

#

# document_text_preprocessed_stemming_links = {}
# document_text_preprocessed_lemmatize_links = {}
#
# for doc_dict in doc_links:
#     for doc_name, doc_id in doc_dict.items():
#         stemming_text = preprocessed_stemming.get(str(doc_id), "Текст не найден")
#         lemmatize_text = preprocessed_lemmatize.get(str(doc_id), "Текст не найден")
#
#         document_text_preprocessed_stemming_links[doc_name] = stemming_text
#         document_text_preprocessed_lemmatize_links[doc_name] = lemmatize_text
#
# with open('document_text_preprocessed_stemming_links.json', 'w', encoding='utf-8') as file:
#     json.dump(document_text_preprocessed_stemming_links, file, indent=4, ensure_ascii=False)
#
# with open('document_text_preprocessed_lemmatize_links.json', 'w', encoding='utf-8') as file:
#     json.dump(document_text_preprocessed_lemmatize_links, file, indent=4, ensure_ascii=False)

# key_counts = defaultdict(int)
#
# for item in data:
#     for key in item.keys():
#         key_counts[key] += 1
#
# print("Количество объектов для каждого ключа:")
# for key, count in key_counts.items():
#     print(f"Ключ '{key}': {count}")
#
# print("\nКлючи, которые встречаются более одного раза:")
# for key, count in key_counts.items():
#     if count > 1:
#         print(f"Ключ '{key}': {count}")




# # 1. Загрузка данных из JSON
# with open('document_text_preprocessed_lemmatize.json', 'r', encoding='utf-8') as file:
#     preprocessed = json.load(file)
#
# # 2. Список фраз для удаления
# phrases_to_remove = [
#     "получить услуга", "какой документ нужный", "стоимость услуга порядок оплата",
#     "результат получить", "срок хранение результат", "получить результат",
#     "описание результат", "срок оказание услуга", "основание отказ",
#     "основание приостановление услуга", "основание отказ приём документ",
#     "подать заявление", "информация формирование сертификат электронный вид",
#     "часто задавать вопрос", "нормативный документ услуга"
# ]
#
# # 3. Очистка текста от фраз
# def remove_phrases(texts, phrases):
#     cleaned_texts = []
#     for text in texts:
#         text_str = " ".join(text)
#         for phrase in phrases:
#             text_str = text_str.replace(phrase, '')
#         cleaned_texts.append(text_str.split())
#     return cleaned_texts
# # 4. Очистка текста от нежелательных фраз
# texts = remove_phrases([text for text in preprocessed.values()], phrases_to_remove)
#
#
# # 6. Создание словаря и корпуса для LDA
# dictionary = corpora.Dictionary(texts)
# corpus = [dictionary.doc2bow(text) for text in texts]
#
# # 7. Создание LDA-модели
# num_topics = 5
# lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=20, alpha='auto', random_state=42)
#
# # 8. Вывод тем с ключевыми словами
# print("Темы с ключевыми словами:")
# for i, topic in lda_model.show_topics(num_topics=num_topics, num_words=20, formatted=False):
#     print(f"\nТема {i + 1}:")
#     for word, prob in topic:
#         print(f"{word}: {prob:.3f}")
#
#
#

with open('document_text.json', 'r', encoding='utf-8') as file:
    document_text = json.load(file)

with open('doc_links1.json', 'r', encoding='utf-8') as file:
    doc_links = json.load(file)

with open('document_text_preprocessed_stemming.json', 'r', encoding='utf-8') as file:
    preprocessed_stemming = json.load(file)

with open('document_text_preprocessed_lemmatize.json', 'r', encoding='utf-8') as file:
    preprocessed_lemmatize = json.load(file)

rows = []


def cleaning_text(text):
    # Проверка, является ли текст строкой
    if isinstance(text, str):
        return text.replace('[', '').replace(']', '').replace(',', '')
    else:
        return text


for doc_dict in doc_links:
    for doc_name, doc_id in doc_dict.items():
        original_text = document_text.get(str(doc_id), "Текст не найден")
        lemmatized_text = preprocessed_lemmatize.get(str(doc_id), "Текст не найден")
        stemmed_text = preprocessed_stemming.get(str(doc_id), "Текст не найден")


        original_text = cleaning_text(original_text)
        lemmatized_text = cleaning_text(lemmatized_text)
        stemmed_text = cleaning_text(stemmed_text)

        rows.append([doc_id, doc_name, original_text, lemmatized_text, stemmed_text])

df = pd.DataFrame(rows, columns=['id', 'doc_name', 'text', 'lemm_text', 'stemm_text'])

df.to_csv('documents.csv', index=False, encoding='utf-8', sep=';')

df_new = pd.read_csv('documents.csv', sep=';', encoding='utf-8-sig')

print(df_new['id'])
