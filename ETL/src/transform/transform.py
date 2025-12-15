import os
import numpy as np
from PIL import Image
import shutil

class Transform:
    # ---------------------------------------------
    # 1. Trouver toutes les images RAW : equivalent to bash : find data/raw -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \)
    # ---------------------------------------------
    @staticmethod
    def get_images_list(directory="data/raw"):
        image_extensions = ('.png', '.jpg', '.jpeg')
        image_paths = []

        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))

        return image_paths
    
    # ---------------------------------------------
    # 2. Charger une image depuis path -> Image object (numpy array float32)
    # ---------------------------------------------
    @staticmethod
    def load_image(image_path):
        try:
            # le L mode convertit en noir et blanc (grayscale)
            img = Image.open(image_path).convert("L")   # grayscale
            return np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            raise RuntimeError(f"Erreur chargement image {image_path}: {e}")

    # ---------------------------------------------
    # 3. Ajouter speckle multiplicatif, L = nombre de looks
    # ---------------------------------------------
    @staticmethod
    def add_speckle(img, L=1):
        # bruit Gamma(L,L)
        noise = np.random.gamma(L, 1.0 / L, img.shape).astype(np.float32)
        return img * noise

    # ---------------------------------------------
    # 4. Sauvegarder image (original ou noisy) dans processed/, en gardant le même path
    # ---------------------------------------------
    @staticmethod
    @staticmethod
    def save_processed_image(img, original_path, mode="noisy", L=1):
    # nouveau chemin → dossier par niveau L
      new_path = original_path.replace("raw", f"processed/L{L}")

    # suffixe clean/noisy
      if mode == "noisy":
        new_path = new_path.replace(".png", "_noisy.png")
        new_path = new_path.replace(".jpg", "_noisy.jpg")
        new_path = new_path.replace(".jpeg", "_noisy.jpeg")
      else:
        new_path = new_path.replace(".png", "_clean.png")
        new_path = new_path.replace(".jpg", "_clean.jpg")
        new_path = new_path.replace(".jpeg", "_clean.jpeg")

      os.makedirs(os.path.dirname(new_path), exist_ok=True)

      img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
      Image.fromarray(img_uint8).save(new_path)

      return new_path


    # ---------------------------------------------
    # 5. Orchestration complète pour 1 image
    # ---------------------------------------------
    @staticmethod
    def process_one_image(image_path, L=1):
        try:
            # load
            img = Transform.load_image(image_path)

            # noise
            noisy = Transform.add_speckle(img, L=L)

            # save clean
            clean_path = Transform.save_processed_image(img, image_path, mode="clean")

            # save noisy
            noisy_path = Transform.save_processed_image(noisy, image_path, mode="noisy")

            return clean_path, noisy_path

        except Exception as e:
            return f"Erreur traitement {image_path}: {e}"
        
    # ---------------------------------------------
    # 6. Orchestration complète pour toutes les images dans un dossier
    # ---------------------------------------------
    @staticmethod
    def process_all_images(directory, L=1):
        image_paths = Transform.get_images_list(directory)
        results = []

        for img_path in image_paths:
            result = Transform.process_one_image(img_path, L=L)
            results.append(result)

        return results
    
    # ---------------------------------------------
    # 7. Générer des patches à partir d'une image
    # ---------------------------------------------
    @staticmethod
    def generate_patches(img, patch_size=64, stride=32):
        patches = []
        h, w = img.shape

        # range(start, end, step)
        # start en 0, end en h - patch_size + 1 pour inclure le dernier patch possible
        # step = stride, a chaque itération on avance de 'stride' pixels

        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = img[i:i + patch_size, j:j + patch_size]
                patches.append(patch)

        return patches
    
    # ---------------------------------------------
    # 8. Enregistrer les patches de chaque dataset dans des dossiers séparés
    # ---------------------------------------------
    @staticmethod
    def save_patches(patches, base_path, mode="noisy", L=1):
     new_path = base_path.replace("raw", f"patches/L{L}/{mode}")
     new_path = os.path.dirname(new_path)
     os.makedirs(new_path, exist_ok=True)

     existing_files = os.listdir(new_path)
     start_index = len(existing_files)
     saved_paths = []

     for idx, patch in enumerate(patches):
        patch_path = os.path.join(new_path, f"patch_{start_index + idx}.png")
        patch_uint8 = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(patch_uint8).save(patch_path)
        saved_paths.append(patch_path)

     return saved_paths

    
    # ---------------------------------------------
    # 9. Orchestration complète pour générer et sauvegarder les patches
    # ---------------------------------------------
    @staticmethod
    @staticmethod
    def process_patches_for_image(image_path, patch_size=64, stride=32, L=1):
     try:
        clean_path, noisy_path = Transform.process_one_image(image_path, L=L)
        clean_img = Transform.load_image(clean_path)
        noisy_img = Transform.load_image(noisy_path)

        Transform.save_processed_image(clean_img, image_path, mode="clean", L=L)
        Transform.save_processed_image(noisy_img, image_path, mode="noisy", L=L)

        clean_patches = Transform.generate_patches(clean_img, patch_size=patch_size, stride=stride)
        noisy_patches = Transform.generate_patches(noisy_img, patch_size=patch_size, stride=stride)

        clean_patch_paths = Transform.save_patches(clean_patches, image_path, mode="clean", L=L)
        noisy_patch_paths = Transform.save_patches(noisy_patches, image_path, mode="noisy", L=L)

        return clean_patch_paths, noisy_patch_paths

     except Exception as e:
        return f"Erreur traitement patches pour {image_path}: {e}"

    @staticmethod
    def generate_patches_for_all_images(directory, patch_size=64, stride=32, L=1):
        image_paths = Transform.get_images_list(directory)
        results = []

        # delete existing patches folders if they exist
        dir_to_clean = ["patches", "processed"]

        for dir_name in dir_to_clean:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
        i = 0
        for img_path in image_paths:
            result = Transform.process_patches_for_image(img_path, patch_size=patch_size, stride=stride, L=L)
            results.append(result)
            i += 1
            print(f"Processed {i} / {len(image_paths)} images")

        return results
    
    @staticmethod
    def generate_all_L_levels(directory, patch_size=64, stride=32, levels=[1,2,4,8]):
      for L in levels:
        print(f"\n=== GENERATING DATASET FOR L = {L} ===\n")

        # Effacer uniquement les dossiers de ce L
        for folder in [f"patches/L{L}", f"processed/L{L}"]:
            if os.path.exists(folder):
                shutil.rmtree(folder)

        Transform.generate_patches_for_all_images(
            directory, patch_size=patch_size, stride=stride, L=L
        )
