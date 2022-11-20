import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pymorphy2
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from textblob import TextBlob, Word
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word): #Wordnet Lemmatizer с соответствующим POS-тегом
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_with_postag(sentence): #TextBlob Lemmatizer с соответствующим POS-тегом (часть речи)
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a',
                "N": 'n',
                "V": 'v',
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)

def get_idf(arr):
    N = arr.shape[0]
    res = np.copy(arr)
    df = np.count_nonzero(res, axis=0)
    return np.log((N + 1) / (df + 1)) + 1

def preprocess_sentence(x):
    new_x = re.sub(r'[^\w\s]', ' ', x)  # удаляем знаки препинания
    tokens = word_tokenize(new_x)  # токенизируем
    tokens = [token.lower() for token in tokens]  # меняем Заглавные на lower
    tokens = [i for i in tokens if (i not in stopwords.words('english'))]

    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens] #Wordnet Lemmatizer с соответствующим POS-тегом
    #tokens = [lemmatize_with_postag(token) for token in tokens] # TextBlob Lemmatizer
    return ' '.join(tokens)




if __name__ == '__main__':
    df = pd.read_csv('data/spam.csv', usecols=[0, 1], encoding='latin-1')
    # print(data.head())

    # Преобразовали ham и spam в 0 и 1 соответственно
    le = LabelEncoder()
    df['v1'] = le.fit_transform(df['v1'])


    # print(df)
    # df2 = df.head(2500)
    print(preprocess_sentence("better"))

    #print(lemmatizer.lemmatize('better', pos ="a"))

    df['v2'] = df['v2'].apply(preprocess_sentence)

    x_train, x_test, y_train, y_test = train_test_split(df['v2'].values, df['v1'].values, test_size=0.33,
                                                        random_state=42)
    vectorizer = CountVectorizer(ngram_range=(2, 2))

    x_train = vectorizer.fit_transform(x_train).toarray()

    indices_row_null = np.where(np.all(x_train == 0, axis=1))
    x_train = np.delete(x_train, indices_row_null, axis=0)
    y_train = np.delete(y_train, indices_row_null, axis=0)

   # print('{1}-grams: {0}'.format(vectorizer.get_feature_names(), 2)) вывод токенов в виде ngram = 2

    x_test = vectorizer.transform(x_test).toarray()
    #x_test = vectorizer.transform(x_test).todense()
    get_idf(x_train)

    print(df)



    # TF Количество раз, которое слово встречается в документе, деленное на общее количество слов в документе.

    # набор слов в виде вектора признаков (совершаем переход от набора слов к набору чисел)
    # самый простой способ - работа со словарем
