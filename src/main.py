from preprocess.dataloader import load_raw_data
from preprocess.dataclean import clean_data
from exploritory_analysis.explore import analyse_data
import logging


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s: %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    df = load_raw_data()
    df_clean = clean_data(df)
    analyse_data(df, df_clean)
    del df


if __name__ == '__main__':
    main()
