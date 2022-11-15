import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pymorphy2
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

morph = pymorphy2.MorphAnalyzer()

def preprocess_sentence(x):
    new_x = re.sub(r'[^\w\s]', ' ', x)  # удаляем запятые
    tokens = word_tokenize(new_x) # создаем токенизатор
    tokens = [token.lower() for token in tokens] # меняем Заглавные на lower
    tokens = [i for i in tokens if (i not in stopwords.words('english'))]
    tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return ' '.join(tokens)

if __name__ == '__main__':
    df = pd.read_csv('data/spam.csv', usecols=[0, 1], encoding='latin-1')
    #print(data.head())


    # Преобразовали ham и spam в 0 и 1 соответственно
    le = LabelEncoder()
    df['v1'] = le.fit_transform(df['v1'])


    #print(df)
    df2 = df.head(2500)
    df2['v2'] = df2['v2'].apply(preprocess_sentence)
    print(df2)



