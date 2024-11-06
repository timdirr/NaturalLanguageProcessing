import stanza
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re
from tqdm import tqdm
import os

DATA_PATH = "data"
OUTPUT_FILE = os.path.join(DATA_PATH, "conllu_data.conllu")
BATCH_SIZE = 1000


def remove_stopwords(df):
    stop_words = set(stopwords.words('english'))
    nltk.download('stopwords')
    # TODO: remove stopwords
    pass


def tokenize(rows=1000, stopword_removal=False):
    df = pd.read_csv(os.path.join(DATA_PATH, "clean_data.csv"))
    df_first_N = df.iloc[:rows]
    tokenize_pretokenized = False

    if stopword_removal:
        tokenize_pretokenized = True
        remove_stopwords(df_first_N)

    stanza.download('en')
    nlp = stanza.Pipeline(
        'en', processors='tokenize,lemma,pos', verbose=True, use_gpu=True, tokenize_pretokenized=tokenize_pretokenized)

    conllu_batch = []

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

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

    # Remaining rows if the final batch was less than 1k
    if conllu_batch:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.writelines(conllu_batch)
