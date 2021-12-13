from urllib.request import urlopen
import matplotlib.pyplot as plt
import sns as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from IPython.display import display



URL = "https://raw.githubusercontent.com/shilpibhattacharyya/Friends_Analysis/master/friends_dataset.csv"
COLUMNS_TO_KEEP = ['Speaker', 'Text']
DATA_COUNT_TO_LEARN = 50000


def load_data(url, columns_to_keep):
    print("Download starting...")
    csv = urlopen(url)
    print("Download ended")
    data = pd.read_csv(csv, sep=',', header=0, skip_blank_lines=True).dropna()
    data = data[columns_to_keep]
    mask = (data['Text'].str.len() > 80)
    data = data.loc[mask]
    #print(len(data))
    data['SpeakerID'] = data['Speaker'].factorize()[0]
    speaker_id_df = data[['Speaker', 'SpeakerID']].drop_duplicates().sort_values('SpeakerID')
    speaker_to_id = dict(speaker_id_df.values)
    id_to_speaker = dict(speaker_id_df[['SpeakerID', 'Speaker']].values)
    return data, speaker_to_id, id_to_speaker, speaker_id_df



def accuracies():
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(max_iter=10),
        MultinomialNB(),
        LogisticRegression(random_state=0, max_iter=1),
    ]
    CV = 6

    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
      model_name = model.__class__.__name__
      accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
      for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    print(cv_df.groupby('model_name').accuracy.mean())



def percentage():
    X_train, X_test, y_train, y_test = train_test_split(originalData['Text'], originalData['Speaker'], random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = LinearSVC().fit(X_train_tfidf, y_train)
    sum = 0
    all = 0
    for entry in data.values:
        all += 1
        predicted = clf.predict(count_vect.transform([entry[1]]))
        if(predicted == entry[0]):
            sum += 1
    print((sum / all) * 100)


def grams():
    for speaker, speaker_id in sorted(speaker_to_id.items()):
        features_chi2 = chi2(features, labels == speaker_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names_out())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(speaker))
        print("  Most correlated unigrams:\n    . {}".format('\n    . '.join(unigrams[-N:])))
        print("  Most correlated bigrams:\n    . {}".format('\n    . '.join(bigrams[-N:])))



def plot_data(data):
    plt.figure(figsize=(8, 6))
    data.groupby('Speaker').Text.count().plot.bar(ylim=0)
    plt.show()



def linear():
    model = LinearSVC()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size=0.1, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=speaker_id_df.Speaker.values, yticklabels=speaker_id_df.Speaker.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    for predicted in speaker_id_df.SpeakerID:
        for actual in speaker_id_df.SpeakerID:
            if predicted != actual and conf_mat[actual, predicted] >= 10:
                print("'{}' predicted as '{}' : {} examples.".format(id_to_speaker[actual], id_to_speaker[predicted], conf_mat[actual, predicted]))
                display(data.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Text']], )
                print('')

    plt.show()



originalData, speaker_to_id,  id_to_speaker, speaker_id_df = load_data(URL, COLUMNS_TO_KEEP)

data = originalData[:DATA_COUNT_TO_LEARN]
plot_data(data)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=6, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(data.Text).toarray()
labels = data.SpeakerID
#print(features.shape)

#
N = 6
linear()
percentage()
#accuracies()
