# Dataset

## Méthode

Pour réaliser le dataset, nous avons pris des vidéos de nous réalisant les gestes, que nous  avons ensuite découpés en frame, puis redimensionnés pour une largeur de 256px.
Ensuite, nous avons passé l'ensemble de ces images dans AlphaPose pour récuperer un dataset de squelettes correspondants.

## Classes

| Valeur numérique | Label       |
| ---------------- | ----------- |
| 0                | neutre      |
| 1                | decollage   |
| 2                | droite      |
| 3                | gauche      |
| 4                | atterir     |
| 5                | reculer     |
| 6                | rapprocher  |
| 7                | flip        |
| 8                | gear second |
| 9                | fortnite    |

