import gdown
import os
import requests
import zipfile

from globals import DATA_PATH

# URLs for the Kaggle and Google Drive datasets
kaggle_url = "https://www.kaggle.com/api/v1/datasets/download/rajugc/imdb-movies-dataset-based-on-genre"
drive_folder_id = "1NpchamspqFInPZXbv9xPR33luxRVldsO"
drive_file_urls = {
    "clean_data.csv": "https://drive.google.com/uc?id=12pSL_4PxiyqGTQm8mfomHGEbB3Il1yZG",
    "crawl_data.csv": "https://drive.google.com/uc?id=1HAARdniVyolwvwfYjhXTy5EUzd9kYSkE",
    "conllu_data.conllu": "https://drive.google.com/uc?id=1hN0KffOC_GX0SEthafgBEth6bNhFBL1R"
}

# Define paths
kaggle_path = os.path.join(DATA_PATH, "raw")
zip_path = os.path.join(kaggle_path, "archive.zip")

# Ensure base path exists
os.makedirs(kaggle_path, exist_ok=True)
print(f"Folder {kaggle_path} created.")

# Download from Kaggle if archive.zip doesn't exist
response = requests.get(kaggle_url, stream=True)
response.raise_for_status()  # Check for download errors

with open(zip_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)
print("Kaggle dataset download complete.")

# Unzip the downloaded file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(kaggle_path)
print("Extraction complete.")

os.remove(zip_path)
print("Zip file deleted.")

# Download files from Google Drive if they don't already exist
for file_name, file_url in drive_file_urls.items():
    file_path = os.path.join(DATA_PATH, file_name)
    if not os.path.isfile(file_path):
        gdown.download(file_url, file_path, quiet=False)
        print(f"{file_name} downloaded.")
    else:
        print(f"{file_name} already exists. Skipping download.")
