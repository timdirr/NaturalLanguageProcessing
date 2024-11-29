from abc import ABC, abstractmethod
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
            ngram_range : tuple, optional
                The lower and upper boundary of the range of n-values for different n-grams to be extracted.

        '''
        if vectorizer_name not in ['count', 'tf-idf']:
            raise ValueError("Vectorizer not supported")
        self.vectorizer_name = vectorizer_name
        self.kwargs = kwargs
        self.model = self.set_vectorizer(vectorizer_name)

    def fit(self, X, y=None):
        self.model.fit(self.check_X(X))
        return self

    def check_X(self, X):
        if isinstance(X[0], list):
            X_input = [" ".join(x) for x in X]
        else:
            X_input = X
        return X_input

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X, y=None):
        return self.model.fit_transform(self.check_X(X))

    def set_vectorizer(self, vectorizer_name):
        if vectorizer_name == 'count':
            return CountVectorizer(**self.kwargs)
        else:
            return TfidfVectorizer(**self.kwargs)

    def get_feature_names_out(self):
        return self.model.get_feature_names_out()


class WordEmbeddingModel(ABC, BaseEstimator, TransformerMixin):
    '''
    Superclass for word embedding models.
    Pretrained models are not supposed to be retrained:
        https://stackoverflow.com/questions/68298289/fine-tuning-pre-trained-word2vec-model-with-gensim-4-0.
        https://czarrar.github.io/Gensim-Word2Vec/
    When loading from pretrained the model instance does not matter. They yield the same results as it is essentially a
    wrapper around the pretrained model.

    '''

    def __init__(self, train_embeddings=True, **kwargs):
        self.kwargs = kwargs
        self.train_embeddings = train_embeddings
        self.model: Word2Vec = None

    def fit(self, X, y=None):
        '''
        Fits model to the data.
        Parameters:
        ----------
        X : list of str or list of list of str
            List of descriptions.
        '''
        if self.model is None:
            raise ValueError(
                "Model is not loaded. Please provide a pretrained model.")

        if not self.train_embeddings:
            log.info("Not training embeddings as train_embeddings=False")
            return self

        log.info(f"Fitting model")
        if isinstance(X[0], list):
            X_input = X
        else:
            X_input = [x.split() for x in X]

        # check if vocab already exists
        if self.model.wv.key_to_index:
            log.info("Updating existing vocabulary")
            self.model.build_vocab(X_input, update=True)
        else:
            log.info("Building vocabulary")
            self.model.build_vocab(X_input)

        log.info("Training model")
        self.model.train(X_input, total_examples=len(
            X_input), epochs=self.model.epochs)

        return self

    def load_model(self, model: str):
        '''
        Loads the model.
        Inputs:
        ----------
        model : str
            Path to the model or name of the pretrained model.
        '''
        if os.path.exists(model):
            self.load_from_path(model)
        else:
            self.load_pretrained(model)

    @abstractmethod
    def load_from_path(self, path: str):
        '''
        Loads the model from path.
        Inputs:
        ----------
        path : str
            Path of the model.
        '''
        pass

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

    def vectorize(self, sentence: str):
        '''
        Computes the word embeddings for a single description.
        Inputs:
        ----------
        sentence : str
            Description to vectorize.
        '''
        words = sentence.split()
        word_vectors = np.array([self.model.wv.get_vector(word)
                                for word in words if word in self.model.wv])
        if len(word_vectors) == 0:
            return np.zeros(self.model.vector_size)
        return self.aggregate_vectors(word_vectors)

    def aggregate_vectors(self, word_vectors: np.ndarray):
        '''
        Aggregates word vectors to a single vector.
        Inputs:
        ----------
        word_vectors : np.ndarray
            Array of word vectors.
        '''
        # TODO: Implement different aggregation strategies?
        return word_vectors.mean(axis=0)

    def save_vectors(self, path: str):
        '''
        Save vectors to path
        Inputs:
        ----------
        path : str
            Save path.
        '''
        dir_path = os.path.dirname(path)

        assert dir_path.startswith(
            WORD_EMBEDDING_PATH), f"Model path must be in WORD_EMBEDDING_PATH: {WORD_EMBEDDING_PATH}"
        if not os.path.exists(dir_path):
            log.info(f"Directory {dir_path} does not exist")
            os.makedirs(dir_path)
            log.info(f"Directory {dir_path} created")

        log.info(f"Saving vectors to {path}")
        word_vectors = self.model.wv
        word_vectors.save(path)
        log.info("Vectors saved")

    def save(self, path: str):
        '''
        Save model to path
        Inputs:
        ----------
        path : str
            Save path.
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

    def load_pretrained(self, model_name: str):
        '''
        Loads pretrained model.
        Inputs:
        ----------
        model_name : str
            Name of the pretrained model.
        '''
        log.info(f"Loading pretrained model: {model_name}")
        self.model.wv = gensim.downloader.load(model_name)
        log.info(f"Loaded pretrained {model_name} model: {self.model}")
        self.train_embeddings = False

    def available_models(self):
        '''
        Returns available pretrained models.
        '''
        return list(gensim.downloader.info()['models'].keys())

    def get_feature_names_out(self):
        pass


class Word2VecModel(WordEmbeddingModel):
    def __init__(self, model=None, train_embeddings=True, **kwargs):
        super().__init__(train_embeddings=train_embeddings, **kwargs)
        if model:
            self.load_model(model)
        else:
            self.model = Word2Vec(**kwargs)
        self.kwargs = kwargs

    def load_from_path(self, path: str):
        '''
        Loads the model from path.
        Inputs:
        ----------
        path : str
            Path of the model.
        '''
        log.info(f"Loading Word2Vec model from path: {path}")
        self.model = Word2Vec.load(path)
        log.info(f"Loaded Word2Vec model: {self.model}")


class FastTextModel(WordEmbeddingModel):
    def __init__(self, model=None, train_embeddings=True, **kwargs):
        super().__init__(train_embeddings=train_embeddings, **kwargs)
        if model:
            self.load_model(model)
        else:
            self.model = FastText(**kwargs)
        self.kwargs = kwargs

    def load_from_path(self, path):
        '''
        Loads the model from path.
        Inputs:
        ----------
        path : str
            Path of the model.
        '''
        log.info(f"Loading FastText model from path: {path}")
        self.model = FastText.load(path)
        log.info(f"Loaded FastText model: {self.model}")
