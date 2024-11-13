import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import gensim
from gensim.models import Word2Vec
import gensim.downloader

from globals import MODEL_PATH

import logging as log
import gzip


def get_text_model(model_name: str = 'CountVectorizer', **kwargs):
    '''
    Returns a text model
    Inputs:
    model_name: str
        name of the model to use, supported values are 'CountVectorizer', 'TfidfVectorizer', 'Word2Vec'
    kwargs: dict
        parameters for the model
    '''
    models = {
        'CountVectorizer': BagOfWords('count', **kwargs),
        'TfidfVectorizer': BagOfWords('tf-idf', **kwargs),
        'Word2Vec': Word2VecModel(**kwargs),
    }
    log.info(f"Using {model_name} model as text model")
    return models[model_name]


def download_w2v_model(model_name: str, save_dir: os.path = os.path.join(MODEL_PATH, "word2vec")):
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


class Word2VecModel(BaseEstimator, TransformerMixin):
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

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y=None):
        '''
        Fits model to the data.
        Parameters:
        ----------
        X : list of str or list of list of str
            List of descriptions.
        '''
        log.info("Fitting Word2Vec model")
        if isinstance(X[0], list):
            X_input = X
        else:
            X_input = [x.split() for x in X]

        self.model = Word2Vec(X_input, **self.kwargs)

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

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def vectorize(self, sentence):
        words = sentence.split()
        word_vectors = np.array([self.model.wv[word]
                                 for word in words if word in self.model.wv]
                                )
        if len(word_vectors) == 0:
            return np.zeros(100)
        return self.aggregate_vectors(word_vectors)

    def aggregate_vectors(self, word_vectors):
        # Implement different affregation strategies
        return word_vectors.mean(axis=0)

    def load_model(self, model='word2vec-google-news-300'):
        '''
        Load pretrained embeddings
        '''
        if os.path.exists(model):
            if model.endswith('.gz'):
                with gzip.open(model, 'rt', encoding='utf-8') as f:
                    self.model = gensim.models.KeyedVectors.load_word2vec_format(
                        f, binary=False)
            else:
                self.model = gensim.models.KeyedVectors.load_word2vec_format(
                    model, binary=True)
        else:
            log.info("No model path found, loading from gensim")
            try:
                self.model = gensim.downloader.load(model)
                log.info("Pretrained model loaded")
            except ValueError:
                print("Pretrained model not found")
                return None

    def save_model(self, path):
        '''
        Saves model to WORD2VEC_MODEL_PATH
        '''

        dir_path = os.path.dirname(path)
        file_name = os.path.basename(path)

        assert dir_path.startswith(
            MODEL_PATH), "Model path must be in MODEL_PATH"
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"
        log.info(f"Saving model to {path}")
        self.model.wv.save(path)
        log.info("Model saved")

    def available_models(self):
        return list(gensim.downloader.info()['models'].keys())
