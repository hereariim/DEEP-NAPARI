# Notebook Code

## Code

Le code est une inférence d'image pour détecter les noyaux à partir d'un modèle U-NET. Ce modèle U-NET est sauvegardé sous format tensorflow .h5. 

En entrée, un utilisateur écrit:

- model_path_ : Chemin absolu du modèle d'apprentissage
- input_ : Chemin absolu d'une image RGB

En sortie, un utilisateur obtient:

- output_ : Une masque binaire

Algorithme : 

Une image RGB est réduit à la taille d'entrée du modèle d'apprentissage. Ce modèle segmente cette image pour obtenir un masque de probabilité. Pour détecter les noyaux, un seuillage est appliqué pour obtenir un masque binaire. Enfin, ce masque binaire est redimensionné à sa taille d'origine.


## Run Notebook

Aller sur le feuille de calcul 'mifobio.ipynb'.
En haut à droite, connecter le noyau à l'environnement napari.

En entrée, le chemin absolu de model_path_ et input_

Visualiser, le résultat de la segmentation.


