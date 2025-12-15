# HOW TO USE THE ETL SCRIPTS

This directory contains scripts to extract, transform, and load (ETL) SAR data for denoising projects (for each number of looks L) Below are the instructions on how to use these scripts effectively.

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

