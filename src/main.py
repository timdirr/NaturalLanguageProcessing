from preprocess.dataloader import load_raw_data
from preprocess.dataclean import clean_data
import exploritory_analysis.raw_data_exploration as raw_data_exploration
import logging as log


def main():
    log.basicConfig(level=log.INFO,
                    format='%(asctime)s: %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

    df = load_raw_data()
    df_clean = clean_data(df)
    raw_data_exploration.analyse_data(df)
    del df


if __name__ == '__main__':
    main()
