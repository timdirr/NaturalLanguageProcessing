import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import gensim
from gensim.models import Word2Vec, FastText
import gensim.downloader
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_model


from globals import MODEL_PATH, WORD_EMBEDDING_PATH

import logging as log


def download_word_embedding(model_name: str, save_dir: os.path = WORD_EMBEDDING_PATH):
    '''
    Downloads a pretrained Word2Vec model using Gensim's downloader.

    Parameters:
    ----------
    model_name : str
        The name of the model to download (as recognized by Gensim).
    save_dir : str
        The directory where the model should be saved.
    '''
    assert os.path.exists(save_dir), f"Directory {save_dir} does not exist"
    original_base_dir = gensim.downloader.BASE_DIR
    gensim.downloader.BASE_DIR = save_dir  # Set the download directory
    try:
        path = gensim.downloader.load(model_name, return_path=True)
        log.info(f"Model downloaded to {path}")
    except ValueError as e:
        log.error(f"Model '{model_name}' not found. Available models: {
                  gensim.downloader.info()['models'].keys()}")
    finally:
        gensim.downloader.BASE_DIR = original_base_dir  # Reset to original directory


class BagOfWords(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer_name, **kwargs):
        '''
        Initializes the BagOfWords model.

        Parameters:
        ----------
        vectorizer_name : str
            Name of the vectorizer to use. Supported values are 'count' and 'tf-idf'.
        kwargs : dict
            Arguments to pass to the vectorizer.
        '''
        self.vectorizer_name = vectorizer_name
        self.kwargs = kwargs
        self.model = self.set_vectorizer(vectorizer_name)

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X, y=None):
        return self.model.fit_transform(X)

    def set_vectorizer(self, vectorizer_name):
        if vectorizer_name == 'count':
            return CountVectorizer(**self.kwargs)
        elif vectorizer_name == 'tf-idf':
            return TfidfVectorizer(**self.kwargs)
        else:
            raise ValueError("Vectorizer not supported")

    def get_feature_names_out(self):
        return self.model.get_feature_names_out()


class WordEmbeddingModel(BaseEstimator, TransformerMixin):
    '''
    Superclass for word embedding models.
    '''

    def __init__(self, train_embeddings=True, **kwargs):
        self.kwargs = kwargs
        self.model = None
        self.train_embeddings = train_embeddings

    def fit(self, X, y=None):
        '''
        Fits model to the data.
        Parameters:
        ----------
        X : list of str or list of list of str
            List of descriptions.
        '''
        if not self.train_embeddings:
            log.info("Not training embeddings as train_embeddings=False")
            log.info(self.model)
            if self.model is None:
                raise ValueError(
                    "Model is not loaded. Please provide a pretrained model.")
            return self

        log.info(f"Fitting model")
        if isinstance(X[0], list):
            X_input = X
        else:
            X_input = [x.split() for x in X]

        # check if vocab already exists
        if self.model.wv.index_to_key != []:
            log.info("Vocabulary already exists, updating model")
            log.info("Building vocabulary")
            self.model.build_vocab(X_input, update=True)
            log.info("Training model")
            self.model.train(X_input, total_examples=len(
                X_input), epochs=self.model.epochs)
        else:
            log.info("Building vocabulary")
            self.model.build_vocab(X_input)
            log.info("Training model")
            self.model.train(X_input, total_examples=len(
                X_input), epochs=self.model.epochs)

        return self

    def transform(self, X):
        '''
        Transforms data to word embeddings.
        Inputs:
        ----------
        X : list of str
            List of descriptions.

        '''
        if isinstance(X[0], list):
            return np.array([self.vectorize(" ".join(sentence))
                             for sentence in X])
        return np.array([self.vectorize(sentence) for sentence in X])

    def vectorize(self, sentence):
        '''
        Computes the word embeddings for a single description.
        Inputs:
        ----------
        sentence : str
            Description to vectorize.
        '''
        raise NotImplementedError("Subclasses must implement this method")

    def aggregate_vectors(self, word_vectors):
        # Implement different affregation strategies
        return word_vectors.mean(axis=0)

    def save_model(self, path):
        '''
        Save model to path
        '''
        dir_path = os.path.dirname(path)

        assert dir_path.startswith(
            WORD_EMBEDDING_PATH), f"Model path must be in WORD_EMBEDDING_PATH: {WORD_EMBEDDING_PATH}"
        if not os.path.exists(dir_path):
            log.info(f"Directory {dir_path} does not exist")
            os.makedirs(dir_path)
            log.info(f"Directory {dir_path} created")

        log.info(f"Saving model to {path}")
        self.model.save(path)
        log.info("Model saved")

    def available_models(self):
        return list(gensim.downloader.info()['models'].keys())


class Word2VecModel(WordEmbeddingModel):
    '''
    Computes word embeddings for each description.

    Parameters:
    ----------
    vector_size : int, optional
            Dimensionality of the word vectors.
        window : int, optional
            Maximum distance between the current and predicted word within a sentence.
        min_count : int, optional
            Ignores all words with total frequency lower than this.
    '''

    def __init__(self, model=None, train_embeddings=True, **kwargs):
        super().__init__(train_embeddings=train_embeddings, **kwargs)
        if model:
            self.load_model(model)
        else:
            self.model = Word2Vec(**kwargs)
        self.kwargs = kwargs

    def vectorize(self, sentence):
        words = sentence.split()
        word_vectors = np.array([self.model.wv.get_vector(word)
                                for word in words if word in self.model.wv])
        if len(word_vectors) == 0:
            return np.zeros(self.get_vector_size())
        return self.aggregate_vectors(word_vectors)

    def load_model(self, model):
        if os.path.exists(model):
            self.load_from_path(model)
        else:
            self.load_pretrained(model)

    def load_from_path(self, path):
        log.info(f"Loading Word2Vec model from path: {path}")
        self.model = Word2Vec.load(path)
        log.info(f"Loaded Word2Vec model: {self.model}")

    def load_pretrained(self, model_name):
        log.info(f"Loading pretrained Word2Vec model: {model_name}")
        self.model.wv = gensim.downloader.load(model_name)
        log.info(f"Loaded pretrained Word2Vec model: {self.model}")


class FastTextModel(WordEmbeddingModel):
    def __init__(self, model=None, train_embeddings=True, **kwargs):
        super().__init__(train_embeddings=train_embeddings, **kwargs)
        if model:
            self.load_model(model)
        else:
            self.model = FastText(**kwargs)
        self.kwargs = kwargs

    def vectorize(self, sentence):
        words = sentence.split()
        word_vectors = np.array([self.model.wv.get_vector(word)
                                for word in words])
        if len(word_vectors) == 0:
            return np.zeros(self.get_vector_size())
        return self.aggregate_vectors(word_vectors)

    def load_model(self, model):
        if os.path.exists(model):
            self.load_from_path(model)
        else:
            self.load_pretrained(model)

    def load_from_path(self, path):
        log.info(f"Loading FastText model from path: {path}")
        self.model = FastText.load(path)
        log.info(f"Loaded FastText model: {self.model}")

    def load_pretrained(self, model_name):
        log.info(f"Loading pretrained FastText model: {model_name}")
        self.model.wv = gensim.downloader.load(model_name)
        log.info(f"Loaded pretrained FastText model: {self.model}")
