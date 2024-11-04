from preprocess.dataloader import load_raw_data
from preprocess.dataclean import clean_data
import exploritory_analysis.raw_data_exploration as raw_data_exploration
import exploritory_analysis.clean_data_exploration as clean_data_exploration
import logging as log


def main():
    log.basicConfig(level=log.INFO,
                    format='%(asctime)s: %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

    output_path = "data/clean_data.csv"
    # TODO: add command line arguments to main to enable crawling of IMDB api
    # TODO: add command line arguments to main to enable storage of intermediate files needed crawler.
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    df_clean.to_csv(output_path, index=False, quoting=1)

    log.info(f"Cleaned data saved to {output_path}")
    raw_data_exploration.analyse_data(df_raw)
    clean_data_exploration.analyse_data(df_clean)
    del df_raw


if __name__ == '__main__':
    main()

# %%
