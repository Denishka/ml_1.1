import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pymorphy2
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

morph = pymorphy2.MorphAnalyzer()


def preprocess_sentence(x):
    new_x = re.sub(r'[^\w\s]', ' ', x)  # удаляем запятые
    tokens = word_tokenize(new_x)  # создаем токенизатор
    tokens = [token.lower() for token in tokens]  # меняем Заглавные на lower
    tokens = [i for i in tokens if (i not in stopwords.words('english'))]
    tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return ' '.join(tokens)


if __name__ == '__main__':
    df = pd.read_csv('data/spam.csv', usecols=[0, 1], encoding='latin-1')
    # print(data.head())

    # Преобразовали ham и spam в 0 и 1 соответственно
    le = LabelEncoder()
    df['v1'] = le.fit_transform(df['v1'])

    # print(df)
    # df2 = df.head(2500)
    df['v2'] = df['v2'].apply(preprocess_sentence)

    x_train, x_test, y_train, y_test = train_test_split(df['v2'].values, df['v1'].values, test_size=0.33,
                                                        random_state=42)

    vectorizer = CountVectorizer(analyzer='word', stop_words='english', ngram_range=(2, 2))
    x_train = vectorizer.fit_transform(x_train).toarray()
    x_test = vectorizer.transform(x_test).toarray()

    print(df)


    # набор слов в виде вектора признаков (совершаем переход от набора слов к набору чисел)
    # самый простой способ - работа со словарем
