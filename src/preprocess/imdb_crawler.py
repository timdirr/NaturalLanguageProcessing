import threading
import os
import concurrent.futures
import re
import glob

import pandas as pd
import numpy as np

from tqdm import tqdm
from imdb import Cinemagoer

# TODO: revert to crawl/ after debugging
CRAWL_FOLDER = "../crawl/"
class IMDBCrawler():
    def __init__(self, df):
        self.ids = self.__filter_crawl_ids(df)
        self.ia = Cinemagoer()
        self.lock = threading.Lock()

    def __filter_crawl_ids(self, df):
        ids = df[df["description"].str.contains("See full summary") | df["description"].str.contains("Add a Plot")]["movie_id"].unique()
        return ids

    def __find_current_file(self):
        if os.path.exists(f"{CRAWL_FOLDER}crawled_complete.csv"):
            return f"{CRAWL_FOLDER}crawled_complete.csv"

        # TODO: find the highest file in the highest crawl iter/ folder
        pattern = re.compile(r"crawled(\d+)\.csv")
        largest_x = -1
        largest_file = None

        # TODO: remove after debugging
        for file_path in glob.glob(f"../crawl/crawled*.csv"):
            file_name = os.path.basename(file_path)
            match = pattern.match(file_name)
            if match:
                x = int(match.group(1))
                if x > largest_x:
                    largest_x = x
                    largest_file = file_name

        return largest_file

    def crawl(self):
        file = self.__find_current_file()
        max_crawl_iter = 5

        if file is None:
            missing_ids = self.ids
            current_crawl = 1
            while len(missing_ids) > 0 or max_crawl_iter < current_crawl:
                os.mkdir(f"crawl/iter{current_crawl}")
                crawl_dir = f"crawl/iter{current_crawl}/"
                self.__crawl(missing_ids, crawl_dir)
                crawled_movies = self.__find_current_file()
                crawled_df = pd.read_csv(f'{CRAWL_FOLDER}{crawled_movies}')

                missing_ids = np.setdiff1d(np.array(self.ids), crawled_df[~pd.isna(crawled_df["plots"])]['ids'].unique(), assume_unique=True)
                current_crawl += 1
        else:
            crawled_df = pd.read_csv('../crawl/' + file)
            missing_ids = np.setdiff1d(np.array(self.ids), crawled_df[~pd.isna(crawled_df["plots"])]['ids'].unique(), assume_unique=True)

            current_crawl = 1
            while len(missing_ids) > 0 or max_crawl_iter < current_crawl:
                crawl_dir = f"{CRAWL_FOLDER}/iter{current_crawl}/"
                os.mkdir(crawl_dir)
                self.__crawl(missing_ids, crawl_dir)
                crawled_movies = self.__find_current_file()
                crawled_df = pd.read_csv(f'{CRAWL_FOLDER}{crawled_movies}')

                missing_ids = np.setdiff1d(np.array(self.ids), crawled_df[~pd.isna(crawled_df["plots"])]['ids'].unique(), assume_unique=True)
                current_crawl += 1


    def __crawl(self, missing_ids, dir):
        # Shared lists to store results
        ids = []
        plots = []

        def process_movie(id):
            try:
                movie = self.ia.get_movie_plot(id[2:])['data']

                with self.lock:
                    ids.append(id)
                    if "plot" in movie:
                        plots.append(movie['plot'][0])
                    else:
                        plots.append("NaN")

                    if len(ids) % 1000 == 0:
                        df = pd.DataFrame({'ids': ids, 'plots': plots})
                        df.to_csv(os.path.join(os.getcwd(), f'{dir}crawled{len(ids)}.csv'), index=False)
            except Exception:
                pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            list(tqdm(executor.map(process_movie, missing_ids), total=len(missing_ids), desc="Crawling movies"))

        df_final = pd.DataFrame({'ids': ids, 'plots': plots})
        df_final.to_csv(os.path.join(os.getcwd(), f'{dir}crawled_complete.csv'), index=False)

        return df_final



