#1. импорт библиотек
import gensim
from gensim import corpora, models
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from gensim.models import CoherenceModel, LdaModel
from wordcloud import WordCloud

#2. загрузка корпуса и словаря
corpus = corpora.BleiCorpus('ap/ap.dat', 'ap/vocab.txt')

#3. создание модели LDA
model = models.ldamodel.LdaModel(corpus=corpus, id2word=corpus.id2word, num_topics=100, alpha='symmetric')

#3.1 анализ частоты тем, ассоциированных с документами
num_topics_used = [len(model[doc]) for doc in corpus]
plt.hist(num_topics_used)
plt.xlabel('Количество тем')
plt.ylabel('Количество документов')
plt.title('Частота тем в документах')
plt.savefig('topic_frequency.png')
plt.show()

#3.2 вывод 10 тем с ключевыми словами
print("Топ-10 тем с ключевыми словами:")
for ti in range(10):
    words = model.show_topic(ti, 10)
    print(f'Тема {ti}:', words)

#4. предобработка текстов
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = gensim.utils.simple_preprocess(text)
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

texts = [preprocess_text(doc) for doc in open('ap/ap.txt', 'r').readlines()]

#5. оптимизация количества тем с помощью связности, определение качества модели
def evaluate_coherence(dictionary, corpus, texts, start=2, limit=40, step=6):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v', processes=1)
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
model_list, coherence_values = evaluate_coherence(dictionary=dictionary, corpus=corpus, texts=texts, start=2, limit=40, step=6)

#5.1 построение графика связности для оптимизации количества тем
plt.plot(range(2, 40, 6), coherence_values)
plt.xlabel("Количество тем")
plt.ylabel("Связность")
plt.title("Оптимизация количества тем по метрике связности")
plt.savefig('coherence_optimization.png')
plt.show()

#6.0 в виде библиотеки с облаками слов
print("Облако слов для первых 10 тем:")
for i in range(10):
    plt.figure()
    words = dict(model.show_topic(i, 20))
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate_from_frequencies(words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(f'Тема {i}')
    plt.savefig(f'wordcloud_topic_{i}.png')
    plt.show()

#6.1 табличное представление ключевых слов и их частот для первых 10 тем
print("\nТабличное представление ключевых слов для первых 10 тем:")
for i in range(10):
    print(f"\nТема {i + 1}:")
    words = dict(model.show_topic(i, 10))
    print(f"{'Слово':<15} {'Частота':<10}")
    print("-" * 25)
    for word, freq in words.items():
        print(f"{word:<15} {freq:.3f}")


#6.2 линейное представление частоты ключевых слов для первых 10 тем
print("\nЛинейная визуализация частоты ключевых слов для первых 10 тем:")
for i in range(10):
    print(f"\nТема {i + 1}:")
    words = dict(model.show_topic(i, 10))
    for word, freq in words.items():
        line = "=" * int(freq * 50)
        print(f"{word}: {line} ({freq:.3f})")
