from classifier.base import MultiLabelClassifier
import pandas as pd
import os

from globals import DATA_PATH
from sklearn.model_selection import train_test_split
from text_modelling.modelling import BagOfWords
from sklearn.metrics import jaccard_score
import re


def load_data():
    df = pd.read_csv(os.path.join('data', 'clean_data.csv'), converters={
        "encoded_genre": lambda x: re.sub(r"[\[\]']", '', x).split(' ')})
    sampled_df = df.sample(n=5000, random_state=42)

    train_df, test_df = train_test_split(sampled_df, test_size=0.2, random_state=42)
    return train_df["description"], train_df["encoded_genre"], test_df["description"], test_df["encoded_genre"]


def main():
    X_train, y_train, X_test, y_test = load_data()
    model = BagOfWords("tf-idf")
    transformed_data = model.fit_transform(X_train)
    classifier = MultiLabelClassifier("knn", n_neighbors=20, weights='distance')
    classifier = classifier.fit(transformed_data, y_train)

    jaccard = jaccard_score(classifier.compute_target_matrix(y_test), classifier.predict(model.transform(X_test)), average='samples')
    print(jaccard)

    classifier = MultiLabelClassifier("svm", C=2.0, kernel='rbf')
    classifier = classifier.fit(transformed_data, y_train)

    jaccard = jaccard_score(classifier.compute_target_matrix(y_test), classifier.predict(model.transform(X_test)), average='samples')
    print(jaccard)

    classifier = MultiLabelClassifier("bayes")
    classifier = classifier.fit(transformed_data, y_train)

    jaccard = jaccard_score(classifier.compute_target_matrix(y_test), classifier.predict(model.transform(X_test)), average='samples')
    print(jaccard)





if __name__ == "__main__":
    main()
#%%
