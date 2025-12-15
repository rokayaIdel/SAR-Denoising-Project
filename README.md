## Tâche : ETL Multi-L + Débruitage dans le domaine logarithmique avec un débruiteur aveugle (Affinage & Évaluation) – Rokaya Id El Mouedden

### Vue d’ensemble
Ma contribution se concentre sur l’évaluation du comportement d’un **débruiteur aveugle pré-entraîné pour un bruit additif** face au **bruit de speckle SAR pour différents nombres de looks \(L\)**. Afin de rendre le bruit de speckle compatible avec des débruiteurs pour bruit additif, j’ai étendu le pipeline ETL pour générer des jeux de données **multi-\(L\)** et appliqué une **transformation logarithmique (homomorphe)** permettant de passer du domaine d’intensité au domaine logarithmique, où le modèle multiplicatif devient approximativement additif.

Ce travail suit un protocole expérimental en trois étapes :  
(1) modification du pipeline ETL pour générer des jeux de données pickle multi-\(L\),  
(2) adoption d’un débruiteur aveugle pré-entraîné (cadre bruit additif) et reformulation du problème de débruitage SAR dans le domaine logarithmique,  
(3) affinage (fine-tuning) et évaluation à l’aide de métriques standards de qualité d’image (PSNR, SSIM).

---

### Génération de données Multi-L (Modification de l’ETL)
Le pipeline ETL a été modifié afin de générer automatiquement des jeux de données indexés par le nombre de looks \(L \in \{1,2,4,8\}\), permettant d’étudier le comportement du modèle aussi bien en régime de fort speckle (petits \(L\)) qu’en régime de faible speckle (grands \(L\)). Pour chaque valeur de \(L\), le pipeline produit des fichiers pickle stockés dans `ETL/data/pickles/` :

- `processed_L{L}.pkl` : images SAR **complètes**, bruitées et propres, stockées sous forme de tableaux NumPy float32.  
- `patches_L{L}.pkl` : **patchs SAR 64×64**, bruités et propres, stockés sous forme de tableaux NumPy float32.

Le modèle d’intensité SAR est défini par  
\( I = X \cdot N \), avec \( N \sim \Gamma(L,L) \), où \( \mathbb{E}[N]=1 \) et \( \mathrm{Var}(N)=1/L \). Lorsque \(L\) augmente, la loi Gamma converge vers une distribution gaussienne, ce qui motive l’analyse des performances pour plusieurs valeurs de \(L\).

---

### Reformulation dans le domaine logarithmique (Transformation homomorphe)
Les débruiteurs aveugles tels que DnCNN sont conçus pour des modèles de bruit additif \( y = x + \varepsilon \), tandis que le speckle SAR suit un modèle multiplicatif. L’application d’une transformation logarithmique conduit à  
\( \log(I+\epsilon) = \log(X+\epsilon) + \log(N) \),  
convertissant ainsi le speckle multiplicatif en bruit additif dans le domaine logarithmique. Une petite constante \( \epsilon \) est introduite pour assurer la stabilité numérique. Après le débruitage, la transformation inverse est appliquée selon  
\( \hat{X} = \exp(\hat{x}_{\log}) - \epsilon \).  
Cette approche homomorphe permet l’utilisation directe de débruiteurs pour bruit additif sur des données SAR (voir **[REF-B]**).

---

### Débruiteur aveugle pré-entraîné et affinage
Un CNN de débruitage aveugle pré-entraîné sur du bruit additif, suivant les travaux de Kai Zhang et al. (**[REF-A]**), a été utilisé comme base de référence. Bien que la transformation logarithmique rende le bruit additif, la distribution de \( \log(N) \) demeure asymétrique, biaisée et à queues lourdes pour de petites valeurs de \(L\). Afin de réduire ce décalage de distribution, le modèle a été affiné sur des données SAR dans le domaine logarithmique à l’aide de paires de patchs bruités/propre générées par le pipeline ETL modifié.

---

### Protocole d’évaluation
Après le débruitage dans le domaine logarithmique et l’application de la transformation exponentielle inverse, les performances ont été évaluées dans le domaine d’intensité à l’aide du **PSNR (dB)** et du **SSIM**. Les résultats ont été analysés séparément pour chaque valeur de \(L\), permettant de comparer les entrées bruitées et les sorties débruitées après affinage.

---

### Discussion et limites
Les performances de débruitage obtenues sont modérées. Ce comportement s’explique par plusieurs facteurs :  
(i) la distribution du bruit de speckle dans le domaine logarithmique s’écarte fortement de la gaussienne, en particulier pour de petits \(L\) ;  
(ii) la transformation logarithmique introduit une asymétrie et un biais dans les statistiques du bruit ;  
(iii) la transformation inverse exponentielle amplifie les petites erreurs du domaine logarithmique ;  
(iv) l’optimisation dans le domaine logarithmique ne correspond pas directement à l’optimisation du PSNR ou du SSIM dans le domaine d’intensité ;  
(v) le régime de faible \(L\) est intrinsèquement plus difficile en raison de la forte variance du speckle.

---

### Perspectives et améliorations futures
Des améliorations potentielles incluent l’utilisation de transformations de stabilisation de variance plus adaptées (telles que Yeo–Johnson ou des transformations logarithmiques généralisées, voir **[REF-B]**), des fonctions de perte robustes adaptées aux bruits à queues lourdes, ainsi que des modèles profonds spécifiques au SAR ou des modèles déroulés dérivés directement de la vraisemblance Gamma du bruit, en cohérence avec la philosophie optimisation-vers-apprentissage profond de ce projet.

## Références

[REF-A] K. Zhang, W. Zuo, Y. Chen, D. Meng, L. Zhang,  
"Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising",  
*IEEE Transactions on Image Processing*, 2017.

[REF-B] X. Hu, M. Zhu, D. Stanković, Z. Feng, S. Mao, et L. Stanković,  
"SAR Despeckling via Log–Yeo–Johnson Transformation and Sparse Representation",  
*arXiv preprint* arXiv:2412.18121, 2024.
