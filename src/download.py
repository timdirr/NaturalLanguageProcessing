import gdown
import os
import requests
import zipfile
import logging as log


from globals import DATA_PATH

# URLs for the Kaggle and Google Drive datasets
kaggle_url = "https://www.kaggle.com/api/v1/datasets/download/rajugc/imdb-movies-dataset-based-on-genre"
drive_folder_id = "1NpchamspqFInPZXbv9xPR33luxRVldsO"
drive_file_urls = {
    "clean_data.csv": "https://drive.google.com/uc?id=12pSL_4PxiyqGTQm8mfomHGEbB3Il1yZG",
    "crawl_data.csv": "https://drive.google.com/uc?id=1HAARdniVyolwvwfYjhXTy5EUzd9kYSkE",
    "conllu_data.conllu": "https://drive.google.com/uc?id=1cyjpL7dXFr_2WhLIlbNPO-PyHeEZBpHI"
}


def download():
    # Define paths
    kaggle_path = os.path.join(DATA_PATH, "raw")
    zip_path = os.path.join(kaggle_path, "archive.zip")

    # Download Kaggle dataset if it doesn't already exist
    if not os.path.isdir(kaggle_path):
        os.makedirs(kaggle_path, exist_ok=True)

        response = requests.get(kaggle_url, stream=True)
        response.raise_for_status()  # Check for download errors

        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Unzip the downloaded file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(kaggle_path)
        os.remove(zip_path)
        log.info("Kaggle dataset downloaded and extracted.")

    # Download files from Google Drive if they don't already exist
    for file_name, file_url in drive_file_urls.items():
        file_path = os.path.join(DATA_PATH, file_name)
        if not os.path.isfile(file_path):
            gdown.download(file_url, file_path, quiet=False)
            log.info(f"{file_name} downloaded.")

    log.info("All files downloaded.")


if __name__ == "__main__":
    log.basicConfig(level=log.INFO,
                    format='%(asctime)s: %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    download()
