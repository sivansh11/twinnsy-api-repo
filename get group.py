# steps to get groups
# - read userdata from file
# - load models and vectorize user data
# - get kMean group


import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import random


def load_model(file_path):
    import pickle
    file = open(file_path, 'rb')
    return pickle.load(file)


def load(file):
    f = open(file)
    user = json.load(f)
    keys = user.keys()
    df = {}
    for key in keys:
        if key not in df:
            df[key] = [user[key]]
        else:
            df[key].append(user[key])
    df = pd.DataFrame.from_dict(df)
    return df


def convert_to_vector_user(user):
    df = pd.DataFrame(user, index=[0])
    main_data = df.copy(True)
    main_data.drop(labels='userId', inplace=True, axis=1)
    df = main_data

    scaler = load_model('scaler.p')
    df = df[['bio']].join(pd.DataFrame(scaler.transform(df.drop('bio', axis=1)), columns=df.columns[1:], index=df.index))

    vectorizer = load_model('vectorizer.p')
    x = vectorizer.transform(df['bio'])
    df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
    new_df = pd.concat([df, df_wrds], axis=1)
    new_df.drop('bio', axis=1, inplace=True)
    pca = load_model('pca.p')
    df_pca = pca.transform(new_df)

    return df_pca


def get_group(user_file_path):
    """
    :param
        user_file_path (json file path for user as query)
    :return:
        group the user belongs to
    """
    user = load(user_file_path)
    vec = convert_to_vector_user(user)
    k_mean = load_model('k_mean.p')
    print(vec)
    return k_mean.predict(vec)


print(get_group('user.txt'))
