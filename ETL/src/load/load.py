import os
import numpy as np
from PIL import Image
import pickle

class Load:
    @staticmethod
    def get_images_list(directory):
        image_extensions = ('.png', '.jpg', '.jpeg')
        image_paths = []

        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))

        return image_paths
    
    @staticmethod
    def load_image(image_path):
        try:
            # le L mode convertit en noir et blanc (grayscale)
            img = Image.open(image_path).convert("L")   # grayscale
            return np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            raise RuntimeError(f"Erreur chargement image {image_path}: {e}")

    @staticmethod
    def load_processed_images(directory="data/processed"):
        X = []  # noisy
        Y = []  # clean

        files = sorted(Load.get_images_list(directory))

        clean_files = [f for f in files if "_clean" in f]
        noisy_files = [f for f in files if "_noisy" in f]

        # map clean/noisy by removing suffix
        clean_map = {f.replace("_clean", "").split(".")[0]: f for f in clean_files}
        noisy_map = {f.replace("_noisy", "").split(".")[0]: f for f in noisy_files}

        keys = sorted(clean_map.keys())

        for k in keys:
            clean_path = clean_map[k]
            noisy_path = noisy_map[k]

            clean = Load.load_image(clean_path)
            noisy = Load.load_image(noisy_path)

            Y.append(clean)
            X.append(noisy)

        return X, Y
    
    @staticmethod
    def load_patches(directory="data/patches"):
        clean_dir = os.path.join(directory, "clean")
        noisy_dir = os.path.join(directory, "noisy")

        X = []  # noisy
        Y = []  # clean

        # walk sur tous les sous-dossiers clean/
        for clean_path in Load.get_images_list(clean_dir):

                # construire chemin noisy correspondant
                relative_path = os.path.relpath(clean_path, clean_dir)
                noisy_path = os.path.join(noisy_dir, relative_path)

                # cas: le fichier n’existe pas
                if not os.path.exists(noisy_path):
                    print(f"[WARNING] Noisy manquant pour {relative_path}, ignoré.")
                    continue

                # charger les images
                clean = Load.load_image(clean_path)
                noisy = Load.load_image(noisy_path)

                Y.append(clean)
                X.append(noisy)

        return X, Y
    
    # -----------------------------------------------------
    # 1. PROCESSED → générer pickle
    # -----------------------------------------------------
    @staticmethod
    def create_processed_pickle(processed_dir="data/processed",
                                pickle_path="data/pickles/processed.pkl"):

        parent_dir = os.path.dirname(pickle_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        DATA = {}

        files = Load.get_images_list(processed_dir)

        # on ne traite que les fichiers _clean
        clean_files = [f for f in files if "_clean" in os.path.basename(f)]

        for clean_path in clean_files:
            # chemin relatif par rapport à processed_dir
            rel = os.path.relpath(clean_path, processed_dir)
            parts = rel.split(os.sep)   # ["div2k", "lr4", "...", "X4", "xxx_clean.png"]

            filename = parts[-1]
            dirs = parts[:-1]

            # construire chemin noisy correspondant
            noisy_filename = filename.replace("_clean", "_noisy")
            noisy_rel = os.path.join(*dirs, noisy_filename) if dirs else noisy_filename
            noisy_path = os.path.join(processed_dir, noisy_rel)

            if not os.path.exists(noisy_path):
                print(f"[WARNING] Noisy missing for {noisy_rel}, skipped.")
                continue

            # charger clean + noisy
            clean_img = Load.load_image(clean_path)
            noisy_img = Load.load_image(noisy_path)

            # construire dict imbriqué selon la hiérarchie de dossiers
            ref = DATA
            for p in dirs:
                ref = ref.setdefault(p, {})

            # au niveau final : listes alignées
            ref.setdefault("clean", []).append(clean_img)
            ref.setdefault("noisy", []).append(noisy_img)

        # sauvegarde pickle
        with open(pickle_path, "wb") as f:
            pickle.dump(DATA, f)

        print(f"[OK] Pickle processed (hierarchical, aligned) → {pickle_path}")

    # -----------------------------------------------------
    # 2. PATCHES → générer pickle
    # -----------------------------------------------------
    @staticmethod
    def create_patches_pickle(patches_dir="data/patches",
                            pickle_path="data/pickles/patches.pkl"):

        parent_dir = os.path.dirname(pickle_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        DATA = {}

        clean_root = os.path.join(patches_dir, "clean")
        noisy_root = os.path.join(patches_dir, "noisy")

        clean_files = Load.get_images_list(clean_root)

        for clean_path in clean_files:

            # rel = chemin relatif à clean/
            # ex: "bsd68/patch_001.png" ou "div2k/lr4/X4/patch_004.png"
            rel = os.path.relpath(clean_path, clean_root)
            parts = rel.split(os.sep)
            filename = parts[-1]
            dirs = parts[:-1]

            # noisy path = même structure + même nom de fichier
            noisy_path = os.path.join(noisy_root, rel)

            if not os.path.exists(noisy_path):
                print(f"[WARNING] Noisy missing for {rel}, skipped.")
                continue

            clean_img = Load.load_image(clean_path)
            noisy_img = Load.load_image(noisy_path)

            # construire dictionnaire hiérarchique selon les dossiers
            ref = DATA
            for p in dirs:
                ref = ref.setdefault(p, {})

            # append paire alignée
            ref.setdefault("clean", []).append(clean_img)
            ref.setdefault("noisy", []).append(noisy_img)

        # sauvegarde pickle
        with open(pickle_path, "wb") as f:
            pickle.dump(DATA, f)

        print(f"[OK] Pickle patches (hierarchical, aligned) → {pickle_path}")





    # -----------------------------------------------------
    # 3. Load depuis pickle
    # -----------------------------------------------------
    @staticmethod
    def load_processed_from_pickle(pickle_path="data/pickles/processed.pkl"):
        return pickle.load(open(pickle_path, "rb"))

    @staticmethod
    def load_patches_from_pickle(pickle_path="data/pickles/patches.pkl"):
        return pickle.load(open(pickle_path, "rb"))