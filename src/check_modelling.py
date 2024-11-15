import numpy as np
import os
from text_modelling.modelling import download_word_embedding, BagOfWords, Word2VecModel, FastTextModel

from gensim.models import Word2Vec
from globals import MODEL_PATH, WORD_EMBEDDING_PATH, SEED
import logging as log

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB as NaiveBayes
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


log.basicConfig(level=log.INFO,
                format='%(asctime)s: %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

np.random.seed(SEED)
data = [
    ("I absolutely loved the new movie I saw last night!", 1),
    ("The product broke after just one use, very disappointing.", 0),
    ("Fantastic service and friendly staff at the hotel.", 1),
    ("I will never shop at this store again.", 0),
    ("The book was captivating from start to finish.", 1),
    ("Terrible customer support experience.", 0),
    ("I highly recommend this restaurant to all my friends.", 1),
    ("The app keeps crashing; it's useless.", 0),
    ("What a beautiful day to spend at the park!", 1),
    ("The flight was delayed for hours without any explanation.", 0),
    ("My new phone works flawlessly and the battery life is great.", 1),
    ("The meal was cold and lacked flavor.", 0),
    ("I had an amazing time at the concert last night.", 1),
    ("Customer service was rude and unhelpful.", 0),
    ("This is the best coffee I've ever tasted!", 1),
    ("The website is confusing and hard to navigate.", 0),
    ("I felt so relaxed after the spa treatment.", 1),
    ("The wait time at the clinic was ridiculous.", 0),
    ("The delivery was quick and the package was well-protected.", 1),
    ("The instructions were unclear and incomplete.", 0),
    ("My kids loved the new playground.", 1),
    ("The concert tickets were overpriced for the quality.", 0),
    ("Excellent craftsmanship on this piece of furniture.", 1),
    ("The software is full of bugs and glitches.", 0),
    ("I was impressed with the level of detail in the artwork.", 1),
    ("The noise from the construction was unbearable.", 0),
    ("Our vacation was absolutely perfect.", 1),
    ("The car's engine failed after only a week.", 0),
    ("I appreciate the prompt response to my inquiry.", 1),
    ("My experience with the company has been frustrating.", 0),
    ("The scenic views were breathtaking.", 1),
    ("The lecture was boring and hard to follow.", 0),
    ("I'm thrilled with the results of the project.", 1),
    ("My order arrived late and damaged.", 0),
    ("This recipe is delicious and easy to make.", 1),
    ("I regret buying this product.", 0),
    ("The performance was outstanding and memorable.", 1),
    ("The staff ignored me the whole time.", 0),
    ("I'm delighted with my new job.", 1),
    ("The event was poorly organized.", 0),
    ("What an incredible experience!", 1),
    ("I wasted my money on this.", 0),
    ("The new policy is beneficial to all employees.", 1),
    ("The hotel room was dirty and smelled bad.", 0),
    ("I feel healthier after following this diet plan.", 1),
    ("The service at the restaurant was unacceptable.", 0),
    ("I can't wait to visit again!", 1),
    ("The film was a total letdown.", 0),
    ("I found the course to be very informative.", 1),
    ("The device stopped working after a day.", 0),
    ("The staff went above and beyond to help me.", 1),
    ("I wouldn't recommend this to anyone.", 0),
    ("This place has the best desserts in town.", 1),
    ("The manual is difficult to understand.", 0),
    ("I enjoyed every minute of our conversation.", 1),
    ("The room was too small and cramped.", 0),
    ("They offered me a fantastic deal.", 1),
    ("I had high hopes but was severely disappointed.", 0),
    ("This song lifts my spirits every time I hear it.", 1),
    ("The package never arrived.", 0),
    ("I admire their commitment to quality.", 1),
    ("The food made me feel sick.", 0),
    ("The new features are exactly what I needed.", 1),
    ("The project failed due to poor management.", 0),
    ("I love how comfortable these shoes are.", 1),
    ("The lecture was a waste of time.", 0),
    ("The garden is blooming beautifully.", 1),
    ("The staff was not attentive to our needs.", 0),
    ("My experience has been nothing but positive.", 1),
    ("The equipment is outdated and inefficient.", 0),
    ("I felt welcomed and appreciated.", 1),
    ("I am dissatisfied with the lack of support.", 0),
    ("The training session was engaging and helpful.", 1),
    ("The room was noisy, and I couldn't sleep.", 0),
    ("This is my favorite place to relax.", 1),
    ("The price is too high for what you get.", 0),
    ("I can't stop raving about this book.", 1),
    ("The menu options were very limited.", 0),
    ("I'm pleased with how the renovation turned out.", 1),
    ("Customer service did not resolve my issue.", 0),
    ("The team was very professional throughout the process.", 1),
    ("I was put on hold for an hour.", 0),
    ("This is the smoothest ride I've ever had in a car.", 1),
    ("The printer keeps jamming.", 0),
    ("Their generosity made a big difference.", 1),
    ("The screen is flickering and hurts my eyes.", 0),
    ("I feel more confident after attending the workshop.", 1),
    ("The plot of the movie was predictable and dull.", 0),
    ("They have earned my trust and loyalty.", 1),
    ("The fabric shrunk after one wash.", 0),
    ("This tool has improved my productivity.", 1),
    ("I couldn't connect to the Wi-Fi at all.", 0),
    ("Their prompt service saved me time.", 1),
    ("The new update has too many bugs.", 0),
    ("I was moved by the sincerity of the speech.", 1),
    ("The area is unsafe at night.", 0),
    ("I'm grateful for the support I received.", 1),
    ("My complaint was ignored.", 0),
    ("The class exceeded my expectations.", 1),
    ("The package was lost, and they offered no apology.", 0),
]

X = np.array([s[0] for s in data])
y = np.array([s[1] for s in data])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)


def test_word_embeddings():

    # load example dataset from sklearn
    # from sklearn.datasets import fetch_20newsgroups
    # newsgroups_train = fetch_20newsgroups(subset='train')
    # X = newsgroups_train.data

    log.info("Testing Word2Vec")

    model = Word2VecModel(
        min_count=1)
    model.fit(X_train)

    log.info(f"Vocabulary: {model.model.wv.index_to_key}")
    features = model.transform(X_train)
    log.info(f"Feature shape: {features.shape}")
    test_features = model.transform(X_test)
    log.info(f"Test Feature shape: {test_features.shape}")

    model.save_model(os.path.join(WORD_EMBEDDING_PATH, "test_modelw2v"))
    model.load_from_path(os.path.join(
        WORD_EMBEDDING_PATH, "test_model_fasttext", "test_model_fasttext"))
    log.info(f"Feature shape: {features.shape}")

    if np.allclose(features, model.transform(X_train)):
        log.info("Model loaded successfully and features are the same")

    log.info("Loading pretrained model from Gensim")
    model.load_pretrained("glove-wiki-gigaword-50")
    features = model.transform(X)
    log.info(f"Feature shape: {features.shape}")

    if np.allclose(features, model.transform(X_train)):
        log.info("Model loaded successfully and features are the same")

    log.info(f"Available models for Word2Vec: {model.available_models()}")

    ########################################################

    log.info("Testing FastText model")

    model = FastTextModel(vector_size=100, window=5,
                          min_count=1, min_n=3, max_n=6)
    model.fit(X)

    log.info(f"Vocabulary: {model.model.wv.index_to_key}")
    features = model.transform(X_train)
    log.info(f"Feature shape: {features.shape}")
    test_features = model.transform(X_test)
    log.info(f"Test Feature shape: {test_features.shape}")
    model.save_model(os.path.join(
        WORD_EMBEDDING_PATH, "test_model_fasttext", "test_model_fasttext"))

    model.load_from_path(os.path.join(
        WORD_EMBEDDING_PATH, "test_model_fasttext", "test_model_fasttext"))
    log.info(f"Feature shape: {features.shape}")

    if np.allclose(features, model.transform(X)):
        log.info("Model loaded successfully and features are the same")

    log.info("Loading pretrained model from Gensim")
    model.load_pretrained("glove-wiki-gigaword-50")
    features = model.transform(X)
    log.info(f"Feature shape: {features.shape}")

    log.info(f"Available models for FastText: {model.available_models()}")
    log.info("TESTING SIMPLE PIPELINE")

    clf = make_pipeline(
        FastTextModel(model=os.path.join(WORD_EMBEDDING_PATH, "test_model_fasttext",
                      "test_model_fasttext"), train_embeddings=True, min_count=1),
        LogisticRegression()
    )
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    log.info(f"Predictions: {y_pred}")

    ########################################################
    log.info("TEST DOWNLOAD MODELS AND LOADING")

    download_word_embedding("glove-wiki-gigaword-50")


def test_word_embeddings(model, pretrained=None):
    log.info(f"Testing {model.__class__.__name__}")

    model.fit(X_train)

    log.info(f"Vocabulary: {model.model.wv.index_to_key}")

    train_features = model.transform(X_train)
    log.info(f"Feature shape: {train_features.shape}")

    test_features = model.transform(X_test)
    log.info(f"Test Feature shape: {test_features.shape}")

    model.save_model(os.path.join(WORD_EMBEDDING_PATH,
                     f"test_model_{model.__class__.__name__}"))
    model.load_from_path(os.path.join(WORD_EMBEDDING_PATH,
                                      f"test_model_{model.__class__.__name__}"))

    if np.allclose(train_features, model.transform(X_train)):
        log.info("Model loaded successfully and features are the same")
    else:
        log.info("Model loaded successfully but features are different")

    clf = make_pipeline(
        model,
        LogisticRegression()
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    log.info(f"Classification report: \n {
             classification_report(y_test, y_pred)}")

    if pretrained:
        log.info(f"Loading pretrained model from Gensim")
        model.load_pretrained(pretrained)
        test_features_pretrained = model.transform(X_test)
        clf = make_pipeline(
            model,
            LogisticRegression()
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        log.info(f"Classification report: \n {
            classification_report(y_test, y_pred)}")
    return test_features, test_features_pretrained


def run_tests_word_embeddings():

    w2v_feats, w2v_feats_pretrained = test_word_embeddings(Word2VecModel(
        min_count=1),
        pretrained="glove-wiki-gigaword-50")

    ft_feats, ft_feats_pretrained = test_word_embeddings(FastTextModel(vector_size=100, window=5,
                                                                       min_count=1, min_n=3, max_n=6),
                                                         pretrained="glove-wiki-gigaword-50")
    if np.allclose(w2v_feats_pretrained, ft_feats_pretrained):
        log.info("Pretrained w2v and ft models generate same features")
    else:
        log.info("Fitted w2v and ft models generate different features")


def test_bag_of_words(model_name, **kwargs):

    model = BagOfWords(model_name, **kwargs)
    log.info(f"Vectorizer used: {model.model}")
    log.info(f"Ngrams: {model.model.ngram_range}")
    model.fit(X_train)
    features = model.transform(X_train)
    log.info(f"Feature names: {model.get_feature_names_out()[:10]}")
    log.info(f"Number of Feature names: {len(model.get_feature_names_out())}")
    log.info(f"Features shape: {features.shape}")
    log.info(f"Type of features: {type(features)}")
    log.info(f"Features: \n {features[0]}")

    clf = make_pipeline(
        model,
        NaiveBayes()
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    log.info(f"Classification report: \n {
             classification_report(y_test, y_pred)}")


def run_tests_bag_of_words():
    test_bag_of_words("count")
    test_bag_of_words("count", ngram_range=(1, 2))
    test_bag_of_words("tf-idf")
    test_bag_of_words("tf-idf", ngram_range=(1, 2))


def main():
    run_tests_bag_of_words()
    run_tests_word_embeddings()


if __name__ == "__main__":
    main()
