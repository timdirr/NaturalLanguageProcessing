import stanza
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import nltk
import re
import os

DATA_PTH = "data"
OUTPUT_FILE = os.path.join(DATA_PTH, "conllu_data.conllu")
N = 1000
BATCH_SIZE = 1000


def tokenize():
    df = pd.read_csv(os.path.join(DATA_PTH, "clean_data.csv"))
    nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos', verbose=False)
    # Download stopwords from nltk if not already downloaded
    nltk.download('stopwords')
    stanza.download('en')
    stop_words = set(stopwords.words('english'))

    conllu_batch = []
    df_first_N = df.iloc[:N]

    # Tokenize and lemmatize descriptions, and gather words for stopword extension
    all_words = []

    for desc in df_first_N['description']:
        doc = nlp(desc)
        # Collect lemmas of all words
        all_words.extend(
            word.lemma.lower()
            for sentence in doc.sentences for word in sentence.words
            if re.match('\w', word.lemma)  # Only words (no punctuation)
        )

    # n most common words to add to stopwords
    n = 20
    common_words = [word for word in all_words if word.lower()
                    not in stop_words]
    potential_stop_words = Counter(common_words).most_common(n)

    # Top 20 most common words from first 10k rows
    # ('love', 2702), ('man', 2216), ('find', 2088), ('young', 1909), ('take', 1702),
    # ('get', 1698), ('girl', 1591), ('marry', 1541), ('fall', 1517), ('go', 1514),
    # ('father', 1490), ('one', 1321), ('become', 1257), ('daughter', 1247), ('woman', 1243),
    # ('life', 1208), ('make', 1176), ('murder', 1151), ('new', 1149), ('two', 1083)

    # Lets only take words that do not have a genre-neutral
    stop_words_ext = ["get", "take", "go", "make", "become", "one", "two"]
    stop_words.update(stop_words_ext)
    # TODO: remove stopwords after lemmatization?

    print(f"Top {n} most common words to add to stopwords:",
          potential_stop_words)

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    for i, row in enumerate(df_first_N.itertuples(index=True, name='Row'), start=1):
        doc = nlp(row.description)

        # Create the CoNLL-U formatted header for the row
        conllu_str = f"# movie_id = {row.movie_id}\n"
        conllu_str += f"# genre = {row.encoded_genre}\n"
        conllu_str += "{:C}\n\n".format(doc)

        # Add the formatted string to the batch
        conllu_batch.append(conllu_str)

        if i % BATCH_SIZE == 0:
            with open(OUTPUT_FILE, "a") as f:
                f.writelines(conllu_batch)

            conllu_batch = []

    # Remaining rows if the final batch was less than 1k
    if conllu_batch:
        with open(OUTPUT_FILE, "a") as f:
            f.writelines(conllu_batch)
