import os


# Data PATHS
# Folders
DATA_PATH = "data"
CRAWL_FOLDER = "crawl"
SPLIT_FOLDER = "split"

# Stratified Data
TEST_FILE = "test.csv"
TRAIN_FILE = "train.csv"
DEV_FILE = "dev.csv"

# Export PATHS
EXPORT_PATH = "export"

# Config
CONFIG_PATH = "config.json"
MODEL_PATH = "models"
WORD_EMBEDDING_PATH = os.path.join(MODEL_PATH, "word_embeddings")

# Constants
SEED = 42069
UNIQUE_GENRES = ['Action',
                 'Adventure',
                 'Animation',
                 'Biography',
                 'Comedy',
                 'Crime',
                 'Drama',
                 'Family',
                 'Fantasy',
                 'Film-Noir',
                 'History',
                 'Horror',
                 'Music',
                 'Musical',
                 'Mystery',
                 'Romance',
                 'Sci-Fi',
                 'Sport',
                 'Thriller',
                 'War',
                 'Western']
