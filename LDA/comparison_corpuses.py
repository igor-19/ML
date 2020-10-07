import os
import gensim
import numpy as np
import json
import copy

from gensim import corpora, models

path_file = r'data_files/recipes.json'

def path_file_data(path_file):
    path_script = os.path.dirname(__file__)
    return os.path.join(path_script, path_file)

with open(path_file_data(path_file)) as f:
    recipes = json.load(f)

texts = [recipe["ingredients"] for recipe in recipes]
dictionary = corpora.Dictionary(texts)   # составляем словарь
corpus = [dictionary.doc2bow(text) for text in texts]  # составляем корпус документов

np.random.seed(76543)
# здесь код для построения модели:
model_LDA = models.ldamodel.LdaModel(corpus, num_topics=40, passes=5, id2word=dictionary)

topics = model_LDA.show_topics(num_topics=40, num_words=10, formatted=False)
topics

count_ingridients = {'salt': 0, "sugar": 0, "water": 0, "mushrooms": 0, "chicken": 0, "eggs": 0}

dictionary2 = copy.deepcopy(dictionary)

top_tokens = [k for k, v in dictionary2.dfs.items() if v > 4000]

dictionary2.filter_tokens(bad_ids=top_tokens)

# size before and after
dict_size_before = len(dictionary)
dict_size_after = len(dictionary2)

print('size before - ', dict_size_before, '\nsize after  - ', dict_size_after)

corpus2 = [dictionary2.doc2bow(text) for text in texts]

# size corpus before and after
def corpus_size(corpus):
    corpus_size = 0
    for recipe in corpus:
        count = len(set(recipe))
        corpus_size += count
    return corpus_size

corpus_size_before = corpus_size(corpus)
corpus_size_after = corpus_size(corpus2)

np.random.seed(76543)
# здесь код для построения модели:
model_LDA2 = models.ldamodel.LdaModel(corpus2, num_topics=40, passes=5, id2word=dictionary2)

# mean coherece
def mean_coherence(top_topics):
    mean_coherence = []
    for v in top_topics:
        mean_coherence.append(v[1])
    return np.mean(mean_coherence)

coherence = mean_coherence(model_LDA.top_topics(corpus))
coherence2 = mean_coherence(model_LDA2.top_topics(corpus2))

print(np.mean([v[1] for v in model_LDA.top_topics(corpus)]))
print(np.mean([v[1] for v in model_LDA2.top_topics(corpus2)]))

def save_answers3(coherence, coherence2):
    path_file_answer = os.path.join(os.path.dirname(__file__), 'cooking_LDA_pa_task3.txt')
    
    with open(path_file_answer, "w") as fout:
        fout.write(" ".join(["%3f"%el for el in [coherence, coherence2]]))

save_answers3(coherence, coherence2)