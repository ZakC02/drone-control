# Projet PAF - Drone contrôlé par des gestes

Ce dépôt contient le code source et les ressources nécessaires pour le projet PAF (Projet d'Application Final) de Télécom Paris, qui consiste à développer un drone contrôlé par des gestes.

## Depuis la mise à jour du projet avec Mediapipe

Pour tester le classificateur, il suffit de lancer `rewrite.py`, et pour utiliser le projet dans son ensemble en utilisant un ordinateur au lieu de la Jetson, il faut lancer `Pilotv2.py` après s'être connecté au WiFi du drone Tello.  
Le dépôt contient également la matrice de confusion de notre classificateur `Confusion_Matrix2.png` et la courbe d'apprentissage pour 200 *epochs*, `training_curve.png`.

---


## Description

Le projet PAF vise à concevoir un système permettant de contrôler un drone à l'aide de gestes. Le drone peut être utilisé dans diverses applications, telles que la surveillance, l'inspection ou le divertissement. Le contrôle par gestes offre une interaction intuitive et sans contact, ce qui rend le drone facile à piloter.

Ce dépôt contient les éléments suivants :

- `Notebooks/` : Contient des notebooks de tests, crétaion de dataset et modèles.

## Dataset

Le jeu de données d'images utilisé pour la classification des gestes peut être trouvé dans le répertoire suivant :

[Google Drive - Dataset des gestes](https://drive.google.com/drive/folders/1XUJaZuG0i3bU9YkrffDBqSbyEIbv4aLF?usp=drive_link)

Le dataset contient un ensemble d'images labélisées représentant différents gestes. Ces images sont utilisées pour l'entraînement et la validation des modèles de classification des gestes.

## Configuration requise

- Python 3.x
- Bibliothèques Python supplémentaires (voir le fichier `requirements.txt`)

## Installation

1. Clonez ce dépôt sur votre machine locale :
git clone https://gitlab.telecom-paris.fr/PAF/2223/drone-geste.git

2. Installez les dépendances nécessaires :
pip install -r requirements.txt

3. Lancez le projet :
#TODO

## Auteurs

- Maxime Girard
- Ronan Lebas
- Zakaria Chahboune
- Samuel Oussou
