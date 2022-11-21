import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
import pymorphy2
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from spacy.lang.en import English
from textblob import TextBlob, Word
from sklearn import metrics

#
# spacy.load('en_core_web_sm')
# nlp = spacy.load('en_core_web_lg')
# en = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):  # Wordnet Lemmatizer с соответствующим POS-тегом
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_with_postag(sentence):  # TextBlob Lemmatizer с соответствующим POS-тегом (часть речи)
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
    df = np.count_nonzero(arr, axis=0)
    return np.log((1 + N) / (1 + df)) + 1

def get_tf_idf(idf, arr):
    return idf*arr


def preprocess_sentence(x):
    new_x = re.sub(r'[^\w\s]', ' ', x)  # удаляем знаки препинания
    tokens = word_tokenize(new_x)  # токенизируем
    tokens = [token.lower() for token in tokens]  # меняем Заглавные на lower
    tokens = [i for i in tokens if (i not in stopwords.words('english'))]
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in
              tokens]  # Wordnet Lemmatizer с соответствующим POS-тегом
    # tokens = [lemmatize_with_postag(token) for token in tokens] # TextBlob Lemmatizer
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

    # print(lemmatizer.lemmatize('better', pos ="a"))

    df['v2'] = df['v2'].apply(preprocess_sentence)

    x_train, x_test, y_train, y_test = train_test_split(df['v2'].values, df['v1'].values, test_size=0.33,
                                                        random_state=42)
    # Bag of Words
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    x_train = vectorizer.fit_transform(x_train).toarray()
    x_test = vectorizer.transform(x_test).toarray()

    indices_row_null = np.where(np.all(x_train == 0, axis=1))
    x_train = np.delete(x_train, indices_row_null, axis=0)
    y_train = np.delete(y_train, indices_row_null, axis=0)

   # print('{1}-grams: {0}'.format(vectorizer.get_feature_names(), 2)) #вывод токенов в виде ngram = 2

    # вычисление tf-idf
    idf = get_idf(x_train)
    X_train = get_tf_idf(idf, x_train)
    x_test = get_tf_idf(idf, x_test)

    model = SGDClassifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)

    # TN = confusion_matrix[1, 1]
    # TP = confusion_matrix[0, 0]
    # FP = confusion_matrix[0, 1]
    # FN = confusion_matrix[1, 0]

    # print(f"Accuracy: {(TP + TN) / (TP + FN + TN + FP)}")
    # print(f"Precision: {TP / (TP + FP)}")
    # print(f"Recall: {TP / (TP + FN)}")

    y_pred_proba = model.decision_function(x_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    # create ROC curve
    plt.plot(fpr, tpr, label = "auc="+str(auc))
    plt.legend(loc=4)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plot_confusion_matrix(model, x_test, y_test)
    plt.show()

