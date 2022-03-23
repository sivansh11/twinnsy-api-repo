# steps to create model:
# - read json file
# - create df
# - convert df to vector
# - try different cluster values
# - clusterize df to fit with n_clusters
# - save k_mean, scaler, vectorizer, pca models


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
random.seed(0)


stopword = stopwords.words('english')
stopword = set(stopword)
lemmatizer = WordNetLemmatizer()


# tunable
cluster_counts = [i for i in range(2, 20, 1)]


# tokenize and lematize bio
def tokenize(text):
    text = text.lower()
    text = text.replace('.', '')
    text = text.split(' ')
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopword]
    return text


# reads file and returns dataframe object
def load(file):
    f = open(file)
    users = json.load(f)
    keys = users[0].keys()
    df = {}
    for user in users:
        for key in keys:
            if key not in df:
                df[key] = [user[key]]
            else:
                df[key].append(user[key])
    df = pd.DataFrame.from_dict(df)
    return df


def convert_to_vector_df(df):
    """
        takes in dataframe object and vectorizes it
        :param
            df
        :returns
            dataframe
            scaler model
            vectorizer model
            pca model
    """
    main_data = df.copy(True)
    main_data.drop(labels='userId', inplace=True, axis=1)
    df = main_data

    # scaler fit transform
    scaler = MinMaxScaler()
    df = df[['bio']].join(pd.DataFrame(scaler.fit_transform(df.drop('bio', axis=1)), columns=df.columns[1:], index=df.index))

    # vectorize fit transform
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(df['bio'])

    df_wrds = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
    df = pd.concat([df, df_wrds], axis=1)

    df.drop('bio', axis=1, inplace=True)

    # pca test to get n_components
    pca = PCA()
    pca.fit_transform(df)

    total_explained_variance = pca.explained_variance_ratio_.cumsum()
    n_over_95 = len(total_explained_variance[total_explained_variance >= .95])
    n_to_reach_95 = df.shape[1] - n_over_95

    # pca fit transform
    pca = PCA(n_components=n_to_reach_95)
    df_pca = pca.fit_transform(df)

    return df_pca, scaler, vectorizer, pca


def try_num_clusters(df):
    """
    :param
        df
    :return
        s_scores (silhouette score for Kmean clustering for i clusters)
    """
    s_scores = []
    for i in cluster_counts:
        print(i)
        k_means = KMeans(n_clusters=i)

        k_means.fit(df)

        cluster_assesment = k_means.predict(df)

        s_scores.append(silhouette_score(df, cluster_assesment))

    return s_scores


def clusterize(df, n_clusters):
    """
    :param
        df
        n_clusters (num clusters taken elbow point of graph)
    :return:
        k_means (model)
    """
    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(df)

    return k_means


def plot_evaluation(y):
    x = cluster_counts

    df = pd.DataFrame(columns=['cluster score'], index=[i for i in range(2, len(y) + 2)])
    df['cluster score'] = y

    plt.figure(figsize=(16, 6))
    plt.style.use('ggplot')
    plt.plot(x, y)
    plt.xlabel('# of clusters')
    plt.ylabel('score')
    plt.show()


def graph_num_cluter_vs_slihouette_score(df):
    s_score = try_num_clusters(df)
    plot_evaluation(s_score)


def save_model(model, file_name):
    import pickle
    file = open(file_name, 'wb')
    pickle.dump(model, file)


df = load('random.txt')
df, scaler, vectorizer, pca = convert_to_vector_df(df)
graph_num_cluter_vs_slihouette_score(df)
k_mean = clusterize(df, 11)

save_model(scaler, 'scaler.p')
save_model(vectorizer, 'vectorizer.p')
save_model(pca, 'pca.p')
save_model(k_mean, 'k_mean.p')
