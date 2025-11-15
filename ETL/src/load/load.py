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

        DATA = {}  # dictionnaire hiérarchique

        files = Load.get_images_list(processed_dir)

        for img_path in files:
            
            # exemple: div2k/lr4/DIV2K_valid_LR_bicubic/X4/img_clean.png
            rel = os.path.relpath(img_path, processed_dir)
            parts = rel.split(os.sep)  # ["div2k", "lr4", "DIV2K_valid...", "X4", "img_clean.png"]

            # clean or noisy ?
            is_clean = "_clean" in parts[-1]
            is_noisy = "_noisy" in parts[-1]

            if not (is_clean or is_noisy):
                continue

            # Load image
            img = Load.load_image(img_path)

            # Build nested dict automatically
            ref = DATA
            for p in parts[:-1]:  # all folders
                ref = ref.setdefault(p, {})

            # Final level: store clean or noisy
            key = "clean" if is_clean else "noisy"
            ref.setdefault(key, []).append(img)

        # Save pickle
        with open(pickle_path, "wb") as f:
            pickle.dump(DATA, f)

        print(f"[OK] Pickle processed (hierarchical) → {pickle_path}")

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

            # relative → bsd68/patch_0.png, or div2k/lr4/.../patch_X.png
            rel = os.path.relpath(clean_path, clean_root)
            parts = rel.split(os.sep)

            # noisy path
            noisy_path = os.path.join(noisy_root, rel)
            if not os.path.exists(noisy_path):
                print(f"[WARNING] Noisy missing for {rel}")
                continue

            clean = Load.load_image(clean_path)
            noisy = Load.load_image(noisy_path)

            # Build nested dict
            ref = DATA
            for p in parts[:-1]:  # folders
                ref = ref.setdefault(p, {})

            ref.setdefault("clean", []).append(clean)
            ref.setdefault("noisy", []).append(noisy)

        with open(pickle_path, "wb") as f:
            pickle.dump(DATA, f)

        print(f"[OK] Pickle patches (hierarchical) → {pickle_path}")






    # -----------------------------------------------------
    # 3. Load depuis pickle
    # -----------------------------------------------------
    @staticmethod
    def load_processed_from_pickle(pickle_path="data/pickles/processed.pkl"):
        return pickle.load(open(pickle_path, "rb"))

    @staticmethod
    def load_patches_from_pickle(pickle_path="data/pickles/patches.pkl"):
        return pickle.load(open(pickle_path, "rb"))