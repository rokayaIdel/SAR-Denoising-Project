# SAR-Denoising-Project
From Optimization to Deep Learning: Unrolled Neural Networks for Image Despeckling
This project implements an unrolled neural network  for SAR image denoising.
The model is trained on image patches generated.The unrolled architecture mimics an iterative optimization algorithm,
where each layer corresponds to one iteration with learned parameters.

# Instructions GitHub : githubInstructions

# Tasks pour contribuer au projet

Data Preparation:
- Collecte et nettoyage des données SAR.
- Préparation des datasets pour l'entraînement et le test.

Tache assignée à : GHAFOULI Mehdi

Output : data/pickles, des fichiers pickle contenant les X, Y pour l'entrainement et le test.
- processed.pkl : renvoient les X, Y avec X : images (completes) bruitées et Y : images (completes) propres.
- patches.pkl : renvoient les X, Y avec X : patches (64x64) bruitées et Y : patches (64x64) propres.
- processed.pkl : vient directement de data/processed + fonction img to numpy (valeurs en float32)
- patches.pkl : vient directement de data/patches + fonction img to numpy (valeurs en float32)
