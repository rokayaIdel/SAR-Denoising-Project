from src.extract.extract import Extract
from src.transform.transform import Transform
from src.load.load import Load
import shutil
import os
if __name__ == "__main__":
    print("=== ETL PROCESS STARTED ===")
    print("")
    print("CLEANING PREVIOUS DATA IF ANY...")
    dirs_to_clean = ["data/processed", "data/patches", "data/pickles", "data/raw"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    print("CLEANING DONE.")
    print("")
    
    print("=== EXTRACT ===")
    Extract.download_datasets()

    print("=== TRANSFORM ===")
    # delete data/processed, data/patches and data/pickles folders if they exist

    dirs_to_clean = ["data/processed", "data/patches", "data/pickles"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    Transform.generate_patches_for_all_images("data/raw", patch_size=64, stride=32, L=5)

    dirs_to_clean = ["data/pickles"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

    print("=== LOAD → PICKLE ===")
    Load.create_processed_pickle()
    Load.create_patches_pickle()

    print("=== DONE ===")
