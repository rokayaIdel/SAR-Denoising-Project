import os
import requests
import urllib.request
import zipfile

class Extract:

    @staticmethod
    def download_bsd68_original(target_dir="data/raw/bsd68"):
        os.makedirs(target_dir, exist_ok=True)

        api_url = "https://api.github.com/repos/clausmichele/CBSD68-dataset/contents/CBSD68/original"

        print("Récupération de la liste des fichiers...")
        response = requests.get(api_url)
        if response.status_code != 200:
            raise Exception("Impossible de récupérer le contenu du dossier GitHub")

        files = response.json()

        for f in files:
            if f["type"] == "file":
                print(f"Téléchargement : {f['name']}")
                file_data = requests.get(f["download_url"]).content

                with open(os.path.join(target_dir, f["name"]), "wb") as out:
                    out.write(file_data)

        print("Téléchargement complet (uniquement original/).")

    @staticmethod
    def download_set12(target_dir="data/raw/set12"):
        os.makedirs(target_dir, exist_ok=True)

        api_url = "https://api.github.com/repos/aGIToz/KerasDnCNN/contents/Set12"

        print("Récupération de la liste des fichiers Set12...")
        response = requests.get(api_url)
        if response.status_code != 200:
            raise Exception("Impossible de récupérer le contenu du dossier Set12 sur GitHub")

        files = response.json()

        for f in files:
            if f["type"] == "file":
                print(f"Téléchargement : {f['name']}")
                file_data = requests.get(f["download_url"]).content

                with open(os.path.join(target_dir, f["name"]), "wb") as out:
                    out.write(file_data)

        print("Téléchargement Set12 terminé.")

    @staticmethod
    def download_div2k_lr4(target_dir="data/raw/div2k/lr4"):
        os.makedirs(target_dir, exist_ok=True)

        urls = [
            "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
            "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip"
        ]

        for url in urls:
            filename = url.split("/")[-1]
            zip_path = os.path.join(target_dir, filename)

            print(f"Téléchargement : {filename}")
            urllib.request.urlretrieve(url, zip_path)

            print(f"Extraction : {filename}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)

            os.remove(zip_path)

        print("DIV2K HR téléchargé et extrait.")



    @staticmethod
    def download_datasets():
        Extract.download_bsd68_original()
        Extract.download_set12()
        #Extract.download_div2k_lr4()