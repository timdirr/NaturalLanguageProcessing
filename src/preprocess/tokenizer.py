import stanza
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re
from tqdm import tqdm
import os
from globals import DATA_PATH, EXPORT_PATH

from concurrent.futures import ThreadPoolExecutor


BATCH_SIZE = 1000
OUTPUT_FILE = os.path.join(DATA_PATH, "conllu_data.conllu")


def remove_stopwords(df):
    stop_words = set(stopwords.words('english'))
    nltk.download('stopwords')
    # TODO: remove stopwords
    pass


def tokenize(rows=None, stopword_removal=False):
    df = pd.read_csv(os.path.join(DATA_PATH, "clean_data.csv"))
    df_first_N = df if rows is None else df.iloc[:rows]

    tokenize_pretokenized = False

    if stopword_removal:
        tokenize_pretokenized = True
        remove_stopwords(df_first_N)

    stanza.download('en')
    nlp = stanza.Pipeline(
        'en', processors='tokenize,lemma,pos', verbose=True, use_gpu=True, tokenize_pretokenized=tokenize_pretokenized)

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    def process_description(row):
        doc = nlp(row['description'])

        # Create the CoNLL-U formatted header for the row
        conllu_str = f"# movie_id = {row['movie_id']}\n"
        conllu_str += f"# genre = {row['encoded_genre']}\n"
        # Formatting borrowed from stanza.utils.conll
        conllu_str += "{:C}\n\n".format(doc)

        return conllu_str

    rows = df_first_N.to_dict('records')

    with ThreadPoolExecutor(max_workers=16) as executor:
        batch = []
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for conllu_str in tqdm(executor.map(process_description, rows), total=len(rows)):
                batch.append(conllu_str)
                if len(batch) >= BATCH_SIZE:
                    f.writelines(batch)
                    batch = []
            if batch:
                f.writelines(batch)

"""
    for i, row in tqdm(enumerate(df_first_N.itertuples(index=True, name='Row'), start=1)):
        doc = nlp(row.description)

        # Create the CoNLL-U formatted header for the row
        conllu_str = f"# movie_id = {row.movie_id}\n"
        conllu_str += f"# genre = {row.encoded_genre}\n"
        # Formatting borrowed from stanza.utils.conll
        conllu_str += "{:C}\n\n".format(doc)

        # Add the formatted string to the batch
        conllu_batch.append(conllu_str)

        if i % BATCH_SIZE == 0:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.writelines(conllu_batch)

            conllu_batch = []
"""

