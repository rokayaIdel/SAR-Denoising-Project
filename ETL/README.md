# HOW TO USE THE ETL SCRIPTS

This directory contains scripts to extract, transform, and load (ETL) SAR data for denoising projects. Below are the instructions on how to use these scripts effectively.

## Prerequisites
Docker, Docker Compose must be installed on your machine. Ensure you have the necessary permissions to run Docker commands.

## Launching the ETL Process
1. Open a terminal and navigate to the `SAR-Denoising-Project/ETL` directory.
2. Run the following command to start the ETL process using Docker Compose:
   ```bash
   docker-compose up --build
   ```
3. The ETL scripts will execute inside the Docker container, processing the SAR data as defined in the `docker-compose.yml` file.
4. Once the process is complete, the processed data will be available in the `data/pickles` directory.

## Output Files
The ETL process generates the following output folders and files:
- `data/pickles/processed.pkl`: Contains the full noisy and clean images as NumPy arrays (float32).
- `data/pickles/patches.pkl`: Contains 64x64 patches of noisy and clean images as NumPy arrays (float32).
- `data/processed`: Directory containing processed full images, the datasets with clean and noisy images.
- `data/patches`: Directory containing 64x64 patches of clean and noisy images.

## TO USE THE PICKLE FILES
```python
import pickle

# file in data/pickles/patches.pkl
with open("data/pickles/patches.pkl", "rb") as f:
	patches = pickle.load(f)

# file in data/pickles/processed.pkl
with open("data/pickles/processed.pkl", "rb") as f:
	processed = pickle.load(f)
```