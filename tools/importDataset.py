import os
import zipfile
import subprocess

# Funzione per scaricare un file e decomprimerlo
def download_and_extract(url, zip_path, extract_path):
    # Scarica il file usando curl
    subprocess.run(["curl", "-L", "-o", zip_path, url])

    # Crea la directory di estrazione se non esiste
    os.makedirs(extract_path, exist_ok=True)

    # Estrai il file ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Elenca i file estratti
    extracted_files = os.listdir(extract_path)
    print(f"Extracted files in {extract_path}:", extracted_files)

# Percorsi principali
base_dir = os.path.expanduser("~/datasets")

# Dataset Train
train_url = "https://zenodo.org/records/5706578/files/Train.zip?download=1"
train_zip_path = os.path.join(base_dir, "Train.zip")
train_extract_path = os.path.join(base_dir, "Train")
download_and_extract(train_url, train_zip_path, train_extract_path)

# Dataset Val
val_url = "https://zenodo.org/records/5706578/files/Val.zip?download=1"
val_zip_path = os.path.join(base_dir, "Val.zip")
val_extract_path = os.path.join(base_dir, "Val")
download_and_extract(val_url, val_zip_path, val_extract_path)

# Dataset Test
test_url = "https://zenodo.org/records/5706578/files/Test.zip?download=1"
test_zip_path = os.path.join(base_dir, "Test.zip")
test_extract_path = os.path.join(base_dir, "Test")
download_and_extract(test_url, test_zip_path, test_extract_path)
