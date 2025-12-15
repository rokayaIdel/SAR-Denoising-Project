## Scripts ETL pour le débruitage SAR Multi-L

Ce répertoire contient des scripts permettant **d’extraire, transformer et charger (ETL)** des données SAR pour des expériences de débruitage, pour **chaque nombre de looks \(L\)**. Ces scripts sont conçus pour générer des jeux de données dans le domaine logarithmique et pour soutenir l’affinage (fine-tuning) ainsi que l’évaluation de modèles profonds de débruitage.

### Structure du répertoire et rôle des scripts

- `create_multil_log_pkl.py`  
  Génère des jeux de données SAR **multi-\(L\)** dans le **domaine logarithmique**.  

- `dataset_log_multiL.py`  
  Définit une **classe de jeu de données compatible PyTorch** pour le chargement des données SAR dans le domaine logarithmique.  
  Elle prend en charge :
  - le chargement des fichiers pickle multi-\(L\),
  - l’appariement des échantillons bruités et propres,
  - la préparation des données pour l’entraînement et la validation.

- `train_finetune.py`  
  Réalise l’**affinage (fine-tuning) d’un débruiteur aveugle pré-entraîné** (par exemple DnCNN) sur des patchs SAR dans le domaine logarithmique.  
  Le script permet :
  - le gel partiel des couches du réseau,
  - l’entraînement sur plusieurs valeurs de \(L\),
  - la sauvegarde des points de contrôle (checkpoints) du modèle affiné.

- `transform/`  
  Contient des utilitaires de transformation auxiliaires utilisés dans le processus ETL.

## Prérequis
Docker et Docker Compose doivent être installés sur votre machine.  
Assurez-vous de disposer des autorisations nécessaires pour exécuter des commandes Docker.

## Lancement du processus ETL
1. Ouvrir un terminal et se placer dans le répertoire `SAR-Denoising-Project/ETL`.
2. Exécuter la commande suivante pour lancer le processus ETL à l’aide de Docker Compose :
   ```bash
   docker-compose up --build
