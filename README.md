# Readme

## Folder Structure

The src/ folder contains all the scripts. To run the main.py, please first create a data/ folder. Inside, create a folder 
raw/ where you copy the .csv files downloaded from https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre?select=history.csv. 

Also, download the crawled data file crawl_data.csv from (TODO add google drive link.) and add it into the data/ folder.

## How the code works

All executions are done via the main.py file, which accepts following command line parameters: 

* --verbose: enable logging
* --preprocess: loads the raw data an cleans it
* --explore: data exploration, with options raw or clean
* --tokenize: tokenization and lemmatization
