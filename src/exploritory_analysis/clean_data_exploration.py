import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging as log
import os
import nltk
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading
import re

from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
from globals import EXPORT_PATH, DATA_PATH
from exploritory_analysis.description_exploration import plot_description_length


def plot_genre_distribution(df: pd.DataFrame):
    log.info("Plotting genre distribution...")
    genre_count = df['genre'].explode().value_counts()

    plt.figure(figsize=(12, 8))
    genre_count.plot(kind='bar')
    plt.title("Genre Distribution")
    plt.xlabel("Genre")
    plt.ylabel("Frequency")
    plt.subplots_adjust(bottom=0.2)

    # add number over bars
    for i, v in enumerate(genre_count):
        plt.text(i, v + 200, str(v), ha='center', rotation=90)

    plt.savefig(os.path.join(EXPORT_PATH, 'genre_distribution.png'))
    log.info("Genre distribution plotted and saved in export folder.")


def plot_cooccurrence_matrix(df: pd.DataFrame):
    log.info("Plotting co-occurrence matrix...")
    genres = np.sort(df['genre'].explode().unique())
    cooccurrence_matrix = np.zeros((len(genres), len(genres)))

    for genre_list in df['genre']:
        for i, genre1 in enumerate(genres):
            if genre1 in genre_list:
                for j, genre2 in enumerate(genres):
                    if genre2 in genre_list:
                        cooccurrence_matrix[i, j] += 1

    # remove self-cooccurrence
    np.fill_diagonal(cooccurrence_matrix, 0)
    plt.figure(figsize=(12, 8))
    plt.imshow(cooccurrence_matrix, cmap='inferno',
               interpolation='nearest')

    plt.colorbar()
    plt.xticks(np.arange(len(genres)), genres, rotation=90)
    plt.yticks(np.arange(len(genres)), genres)
    plt.title("Genre Co-occurrence Matrix")
    plt.savefig(os.path.join(EXPORT_PATH, 'cooccurrence_matrix.png'))
    log.info("Co-occurrence matrix plotted and saved in export folder.")


def plot_most_frequent_combinations(df: pd.DataFrame):
    log.info("Plotting most frequent genre combinations...")
    multi_genre = df[df['genre'].apply(lambda x: len(x) > 1)]
    genre_combinations = multi_genre['genre'].apply(
        lambda x: ', '.join(x)).value_counts()

    plt.figure(figsize=(12, 8))
    genre_combinations.head(20).plot(kind='bar')
    plt.title("Most Frequent Genre Combinations")
    plt.xlabel("Genre Combination")
    plt.ylabel("Frequency")
    plt.subplots_adjust(bottom=0.3)

    # add number over bars
    for i, v in enumerate(genre_combinations.head(20)):
        plt.text(i, v + 200, str(v), ha='center', rotation=90)

    plt.savefig(os.path.join(
        EXPORT_PATH, 'most_frequent_genre_combinations.png'))
    log.info("Most frequent genre combinations plotted and saved in export folder.")

    log.info("Most frequent genre combinations:")
    log.info(genre_combinations.head(20))


def plot_wordcloud(df):
    log.info("Plotting wordclouds per genre...")
    nltk.download('punkt_tab')
    stop_words = set(stopwords.words('english'))
    lock = threading.Lock()

    def preprocess_text(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = nltk.word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in stop_words]
        return tokens

    genre_word_counts = {}

    def word_counter(row):
        genres = row.genre
        description = row.description
        tokens = preprocess_text(description)
        with lock:
            for genre in genres:
                if genre not in genre_word_counts:
                    genre_word_counts[genre] = Counter()
                genre_word_counts[genre].update(tokens)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(word_counter, df.itertuples(index=False)), total=len(df)))

    for genre, word_count in genre_word_counts.items():
        top_words = dict(word_count.most_common(20))

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {genre} Movies")
        plt.tight_layout()
        plt.savefig(os.path.join(EXPORT_PATH, f"wordcloud_{genre}.png"))
        plt.close()
    log.info("WordClouds per genre plotted and saved in export folder.")


def analyse_data(df: pd.DataFrame):

    if not os.path.exists(EXPORT_PATH):
        os.makedirs(EXPORT_PATH)

    plot_genre_distribution(df)
    plot_cooccurrence_matrix(df)
    plot_most_frequent_combinations(df)
    plot_description_length(df, True)
    plot_wordcloud(df)
