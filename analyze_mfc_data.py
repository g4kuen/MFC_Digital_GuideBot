import json
import re
import nltk
from collections import defaultdict
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



with open('data/doc_links_situation_with_theme.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


with open('data/doc_links_situation_with_theme1.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

with open('data/doc_links_situation.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


with open('data/doc_links_situation1.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)


with open('data/document_text_situation.json', 'r', encoding='utf-8') as file:
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

# Ensure necessary NLTK packages are downloaded
nltk.download('punkt')

# Загрузка исходного JSON файла
input_file = 'data/doc_links_situation1.json'
output_file = 'data/doc_links_situation_stemming.json'

with open(input_file, 'r', encoding='utf-8') as file:
    raw_label_data = json.load(file)


processed_data = []

for item in raw_label_data:
    processed_item = {}
    for key, value in item.items():
        original_text = f"{key} {value}"
        stemmed_text = stem_text(original_text)
        lemmatized_text = lemmatize_text(original_text)

        processed_item[key] = stemmed_text

    processed_data.append(processed_item)


with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(processed_data, file, ensure_ascii=False, indent=4)

print(f"Данные успешно обработаны и сохранены в {output_file}")


# preprocessed_data_stemming = {}
# preprocessed_data_lemmatize = {}
#
# for key, values in data.items():
#     print(key)
#     cleaned_values = [clean_text(value) for value in values]
#     cleaned_values = [remove_stopwords(value) for value in cleaned_values]
#
#     stemmed_values = [stem_text(value) for value in cleaned_values]
#     lemmatized_values = [lemmatize_text(value) for value in cleaned_values]
#
#     preprocessed_data_stemming[key] = stemmed_values
#     preprocessed_data_lemmatize[key] = lemmatized_values
#
# with open('document_text_preprocessed_stemming.json', 'w', encoding='utf-8') as file:
#     json.dump(preprocessed_data_stemming, file, indent=4, ensure_ascii=False)
#
# with open('document_text_preprocessed_lemmatize.json', 'w', encoding='utf-8') as file:
#     json.dump(preprocessed_data_lemmatize, file, indent=4, ensure_ascii=False)

# preprocessed_data_stemming = []
# preprocessed_data_lemmatize = []
#
# for key, values in data.items():
#     cleaned_item = {}
#     for value in values:
#         cleaned_key = clean_text(value)
#         cleaned_key = remove_stopwords(cleaned_key)
#         cleaned_item[cleaned_key] = value
#
#     preprocessed_data_stemming.append({stem_text(key): value for key, value in cleaned_item.items()})
#     preprocessed_data_lemmatize.append({lemmatize_text(key): value for key, value in cleaned_item.items()})
#
# with open('document_text_situation_preprocessed_stemming.json', 'w', encoding='utf-8') as file:
#     json.dump(preprocessed_data_stemming, file, indent=4, ensure_ascii=False)
#
# with open('document_text_situation_preprocessed_lemmatize.json', 'w', encoding='utf-8') as file:
#     json.dump(preprocessed_data_lemmatize, file, indent=4, ensure_ascii=False)
#
#
#
#
# # для links
# for item in data.values():
#     if isinstance(item, dict):
#         cleaned_item = {}
#         for key, value in item.items():
#             cleaned_key = clean_text(key)
#             cleaned_key = remove_stopwords(cleaned_key)
#             cleaned_item[cleaned_key] = value
#
#         preprocessed_data_stemming.append({stem_text(key): value for key, value in cleaned_item.items()})
#         preprocessed_data_lemmatize.append({lemmatize_text(key): value for key, value in cleaned_item.items()})
#     else:
#         print(f"Skipping non-dict item: {item}")








#
#
#
#
# document_text_preprocessed_stemming_links = {}
# document_text_preprocessed_lemmatize_links = {}

# for doc_dict in doc_links:
#     for doc_name, doc_id in doc_dict.items():
#         stemming_text = preprocessed_stemming.get(str(doc_id), "Текст не найден")
#         lemmatize_text = preprocessed_lemmatize.get(str(doc_id), "Текст не найден")
#
#         document_text_preprocessed_stemming_links[doc_name] = stemming_text
#         document_text_preprocessed_lemmatize_links[doc_name] = lemmatize_text

# with open('document_text_preprocessed_stemming_links.json', 'w', encoding='utf-8') as file:
#     json.dump(document_text_preprocessed_stemming_links, file, indent=4, ensure_ascii=False)
#
# with open('document_text_preprocessed_lemmatize_links.json', 'w', encoding='utf-8') as file:
#     json.dump(document_text_preprocessed_lemmatize_links, file, indent=4, ensure_ascii=False)
#
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


#
# import json
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# from gensim import corpora, models
#
# # Загрузка данных и стоп-слов
# nltk.download('stopwords')
# nltk.download('punkt')
# stop_words = set(stopwords.words('russian'))
#
# # 1. Загрузка данных из JSON
# with open('document_text_situation_preprocessed_stemming.json', 'r', encoding='utf-8') as file:
#     preprocessed = json.load(file)
#
# # 2. Список фраз для удаления
# phrases_to_remove = [
#     "получ услуг", "какой документ нужн", "стоим услуг порядк оплат",
#     "результ получ", "срок хран результат", "получ результат",
#     "опис результат", "срок оказ услуг", "основан отказ",
#     "основан приостанов услуг", "основан отказ приём документ",
#     "подат заявлен", "информа формирован сертифик электрон вид",
#     "част зада вопрос", "норматив документ услуг"
# ]
#
#
# def remove_phrases_and_stopwords(texts, phrases_to_remove, stop_words):
#     cleaned_texts = []
#     for text in texts:
#         text_str = " ".join(text)
#         filtered_text = text_str
#         for phrase in phrases_to_remove:
#             filtered_text = filtered_text.replace(phrase, '')
#
#         filtered_tokens = word_tokenize(filtered_text)
#
#         final_tokens = [word for word in filtered_tokens if word not in stop_words]
#
#
#         cleaned_texts.append(final_tokens)
#
#     return cleaned_texts
#
#
# # Применяем очистку текста
# texts = remove_phrases_and_stopwords(preprocessed, phrases_to_remove, stop_words)
#
# # 4. Создание словаря и корпуса для LDA
# dictionary = corpora.Dictionary(texts)
# corpus = [dictionary.doc2bow(text) for text in texts]
#
# # 5. Создание LDA-модели
# num_topics = 5
# lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=20, alpha='auto', random_state=42)
#
# # 6. Вывод тем с ключевыми словами
# print("Темы с ключевыми словами:")
# for i, topic in lda_model.show_topics(num_topics=num_topics, num_words=20, formatted=False):
#     print(f"\nТема {i + 1}:")
#     for word, prob in topic:
#         print(f"{word}: {prob:.3f}")
#
# # 7. Создание файла разметки для BERT
# def create_bert_markup(texts, lda_model, dictionary):
#     data = []
#     for text in texts:
#         bow = dictionary.doc2bow(text)  # Преобразование текста в bag-of-words
#         topics = lda_model.get_document_topics(bow)  # Получение вероятностей тем
#         dominant_topic = max(topics, key=lambda x: x[1])[0]  # Тема с наибольшей вероятностью
#         data.append({
#             "text": " ".join(text),
#             "label": dominant_topic
#         })
#     return data
#
# # 8. Генерация разметки и сохранение в файл
# bert_markup = create_bert_markup(texts, lda_model, dictionary)
#
# output_file = 'bert_markup.json'
# with open(output_file, 'w', encoding='utf-8') as file:
#     json.dump(bert_markup, file, ensure_ascii=False, indent=4)
#
# print(f"\nФайл разметки для BERT сохранен как '{output_file}'.")
#
# # 9. Применение разметки BERT к данным из CSV
# csv_file = 'data/documents_situation.csv'
# output_csv = 'bert_markup.csv'
#
# # Загрузка BERT разметки
# with open(output_file, 'r', encoding='utf-8') as file:
#     bert_markup = json.load(file)
#
# # Загрузка данных из CSV
# df = pd.read_csv(csv_file, sep=';')
#
# # Функции очистки и обработки текста
# def remove_phrases_stemmed(text, phrases):
#     for phrase in phrases:
#         text = text.replace(phrase, '')
#     return text.strip()
#
# def cleaning_text(text):
#     if isinstance(text, str):
#         return text.replace('[', '').replace(']', '').replace(',', '')
#     else:
#         return text
#
#
# df['stemm_text_cleaned'] = df['stemm_text'].apply(lambda x: remove_phrases_stemmed(str(x), phrases_to_remove))
# df['stemm_text_cleaned'] = df['stemm_text_cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
#
# rows = []
# for i, entry in enumerate(bert_markup):
#     bert_text = entry['text']
#     bert_label = entry['label']
#
#     # Находим строку в df, соответствующую индексу из bert_markup
#     doc = df.iloc[i]
#
#     # Сохраняем данные в rows
#     doc_id = doc['id']
#     doc_name = doc['doc_name']
#     rows.append({'id': doc_id, 'doc_name': doc_name, 'label': bert_label})
#
# # Создаем DataFrame для сохранения результата
# output_df = pd.DataFrame(rows)
#
# # Сохраняем результат в CSV
# output_csv = 'bert_markup.csv'
# output_df.to_csv(output_csv, sep=';', index=False, encoding='utf-8')
#
# print(f"Файл '{output_csv}' успешно создан.")
#
#
# for index, row in df.head(50).iterrows():
#     print(f"Строка {index + 1}: {row['stemm_text_cleaned']}")
#
# # Печать первых 50 строк из bert_markup
# for index, entry in enumerate(bert_markup[:50]):
#     print(f"Строка {index + 1}:")
#     print(f"Текст: {entry['text']}")

# with open('document_text_situation.json', 'r', encoding='utf-8') as file:
#     document_text = json.load(file)
#
# with open('doc_links_situation1.json', 'r', encoding='utf-8') as file:
#     doc_links = json.load(file)
#
# with open('doc_links_situation_with_theme1.json', 'r', encoding='utf-8') as file:
#     doc_links_with_theme = json.load(file)
#
# rows = []
#
# def cleaning_text(text):
#     if isinstance(text, str):
#         return text.replace('[', '').replace(']', '').replace(',', '')
#     else:
#         return text
#
#
# theme_dict = {}
# for theme, docs in doc_links_with_theme.items():
#     for doc_name, doc_id in docs.items():
#         theme_dict[str(doc_id)] = theme
#
#
# for doc_dict in doc_links:
#     for doc_name, doc_id in doc_dict.items():
#         print(doc_id)
#         original_text = document_text.get(str(doc_id), "Текст не найден")
#
#
#         original_nolist_text=' '.join(original_text)
#         original_nolist_text = clean_text(original_nolist_text)
#         original_nolist_text = remove_stopwords(original_nolist_text)
#         original_nolist_text = cleaning_text(original_nolist_text)
#
#         lemmatized_text = lemmatize_text(original_nolist_text)
#         stemmed_text  = stem_text(original_nolist_text)
#
#         theme = theme_dict.get(str(doc_id), "Тема не найдена")
#
#         rows.append([doc_id, doc_name, original_text, theme, lemmatized_text, stemmed_text])
#         rows.append([doc_id, doc_name, original_text,theme, lemmatized_text, stemmed_text])
#
# df = pd.DataFrame(rows, columns=['id', 'doc_name', 'text', 'theme', 'lemm_text', 'stemm_text'])
# #
# # df.to_csv('documents_situation.csv', index=False, encoding='utf-8', sep=';')
# #
# df_new = pd.read_csv('documents.csv', sep=';', encoding='utf-8')
# #df_new = df_new.drop_duplicates(subset=['id'])
#
# #df_new.to_csv('documents.csv', index=False, encoding='utf-8', sep=';')
# #print(df_new['theme'])
#
# filtered_df = df_new[df_new['theme'].isin(["Федеральная служба судебных приставов России"])]
#
# print(filtered_df['doc_name'])
#
#
#
#
#
# import json
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
# from datasets import Dataset
#
# # 1. Загрузка данных из JSON файла
# with open('document_text_situation.json', 'r', encoding='utf-8') as file:
#     docs_text = json.load(file)
#
# # 2. Объединение всех текстов в один
# combined_text = ' '.join(docs_text)
#
# # 3. Инициализация токенизатора и модели GPT-2
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
#
# # Устанавливаем паддинг-токен
# tokenizer.pad_token = tokenizer.eos_token  # Используем eos_token как паддинг-токен
#
# model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
#
# # 4. Токенизация текста
# inputs = tokenizer(combined_text, return_tensors='pt', max_length=1024, truncation=True, padding='max_length')
#
# # Для обучения GPT-2 мы используем текст как input_ids и labels (метки) как input_ids
# inputs['labels'] = inputs['input_ids']
#
# # 5. Подготовка данных для тренировки
# dataset = Dataset.from_dict({
#     'input_ids': inputs['input_ids'],
#     'attention_mask': inputs['attention_mask'],
#     'labels': inputs['labels']  # добавляем метки в датасет
# })
#
# # 6. Настройка параметров тренировки
# training_args = TrainingArguments(
#     output_dir='./results',          # директория для результатов
#     overwrite_output_dir=True,       # перезаписать директорию
#     num_train_epochs=3,              # количество эпох
#     per_device_train_batch_size=1,   # размер батча
#     save_steps=10_000,               # сохранять модель каждые 10,000 шагов
#     save_total_limit=2,              # сохранять максимум 2 модели
# )
#
# # 7. Создание объекта Trainer
# trainer = Trainer(
#     model=model,                     # модель GPT-2
#     args=training_args,              # параметры тренировки
#     train_dataset=dataset            # тренировочный датасет
# )
#
# # 8. Запуск тренировки
# trainer.train()
#
# # 9. Сохранение модели
# model.save_pretrained('D:/fine_tuned_gpt2')
# tokenizer.save_pretrained('D:/fine_tuned_gpt2')
