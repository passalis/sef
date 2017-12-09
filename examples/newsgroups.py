from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def load_20ng_dataset_bow():

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    # Convert data to tf-idf

    vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.95)
    train_data = vectorizer.fit_transform(newsgroups_train.data)
    test_data = vectorizer.transform(newsgroups_test.data)
    train_data = train_data.todense()
    test_data = test_data.todense()
    train_labels = newsgroups_train.target
    test_labels = newsgroups_test.target

    return train_data, train_labels, test_data, test_labels
