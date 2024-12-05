import json
import re
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pymorphy2
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
# nltk.download('stopwords')
# nltk.download('punkt')


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

def stem_text(text):
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


import json

with open('doc_links1.json', 'r', encoding='utf-8') as file:
    doc_links = json.load(file)

with open('document_text_preprocessed_stemming.json', 'r', encoding='utf-8') as file:
    preprocessed_stemming = json.load(file)

with open('document_text_preprocessed_lemmatize.json', 'r', encoding='utf-8') as file:
    preprocessed_lemmatize = json.load(file)

document_text_preprocessed_stemming_links = {}
document_text_preprocessed_lemmatize_links = {}

for doc_dict in doc_links:
    for doc_name, doc_id in doc_dict.items():
        stemming_text = preprocessed_stemming.get(str(doc_id), "Текст не найден")
        lemmatize_text = preprocessed_lemmatize.get(str(doc_id), "Текст не найден")

        document_text_preprocessed_stemming_links[doc_name] = stemming_text
        document_text_preprocessed_lemmatize_links[doc_name] = lemmatize_text

with open('document_text_preprocessed_stemming_links.json', 'w', encoding='utf-8') as file:
    json.dump(document_text_preprocessed_stemming_links, file, indent=4, ensure_ascii=False)

with open('document_text_preprocessed_lemmatize_links.json', 'w', encoding='utf-8') as file:
    json.dump(document_text_preprocessed_lemmatize_links, file, indent=4, ensure_ascii=False)

# key_counts = defaultdict(int)
#
# for item in data:
#     for key in item.keys():
#         key_counts[key] += 1
#
# # Вывод количества объектов для каждого ключа
# print("Количество объектов для каждого ключа:")
# for key, count in key_counts.items():
#     print(f"Ключ '{key}': {count}")
#
# # Вывод ключей, которые встречаются более одного раза
# print("\nКлючи, которые встречаются более одного раза:")
# for key, count in key_counts.items():
#     if count > 1:
#         print(f"Ключ '{key}': {count}")




# Загрузка данных из файлов JSON
with open('doc_links1.json', 'r', encoding='utf-8') as file:
    doc_links = json.load(file)

with open('document_text.json', 'r', encoding='utf-8') as file:
    document_text = json.load(file)

with open('document_text_preprocessed_lemmatize.json', 'r', encoding='utf-8') as file:
    preprocessed_lemmatize = json.load(file)

with open('document_text_preprocessed_stemming.json', 'r', encoding='utf-8') as file:
    preprocessed_stemming = json.load(file)

# Создание списка для хранения строк DataFrame
rows = []

# Объединение данных
for doc_dict in doc_links:
    for doc_name, doc_id in doc_dict.items():
        # Получаем данные по ID
        original_text = document_text.get(str(doc_id), "Текст не найден")
        lemmatized_text = preprocessed_lemmatize.get(str(doc_id), "Текст не найден")
        stemmed_text = preprocessed_stemming.get(str(doc_id), "Текст не найден")

        # Добавляем строку в список
        rows.append([doc_id, doc_name, original_text, lemmatized_text, stemmed_text])

# Создание DataFrame
df = pd.DataFrame(rows, columns=['id', 'doc_name', 'text', 'lemm_text', 'stemm_text'])

# Сохранение DataFrame в CSV с указанным разделителем
df.to_csv('documents.csv', index=False, encoding='utf-8-sig', sep='`')

print("CSV файл успешно создан с разделителем ';'!")
