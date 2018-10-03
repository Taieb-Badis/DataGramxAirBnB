# The Hackademy

Programme de formation en Data Science basé sur des challenges proposés par des Startup partenaires.

## Prérequis 
Ce document contient une liste de prérequis en terme de programmes informatiques ainsi que des références pour mieux appréhender les différents ateliers.

### Installer Python 3.x (3.4, 3.5 ou 3.6) et pip
**Pour Windows** : <https://www.python.org/downloads/windows/>

**Pour Mac** : <https://www.python.org/downloads/mac-osx/>

ou avec le terminal 
```
brew install python3
brew install python3-pip
```

### Installer Jupyter 
Jupyter est une application web permettant de créer des notebooks, des fichiers mélangeant à la fois du texte (markdown), des images, des équations mathématiques et du code.

Lancer la commande suivante : 
```
 pip3 install jupyter
``` 

### Installer les dépendances usuelles 
Lancer la commande suivante : 
```
pip3 install numpy matplotlib scipy pandas
```

## Configurer son environnement de travail 
Ouvrez un terminal

### Rappel des commandes utiles sur terminal 
- pwd : ou je suis (dans quel dossier je suis) 
- cd *MonRepo* : changer de dossier (direcrtory)
- ls : lister les objets présents (fichier et dossier) dans le répertoire sélectionné
- mkdir *NewRepo* : créer un nouveau dossier de nom *NewRepo*  
- man *UneCommande* : afficher le manuel de la commande sélectionnée

### Créer son dossier de travail 

0 - Ouvrir un terminal 

1 - Créer un dossier (si non existant) *workspace* 

2 - Créer dans le dossier *workspace* un dossier *Ateliers* 

### Importer le repository de GitHub 

0 - Aller sur la page <https://github.com/TheHackademy/DataGramxAirBnB>

1 - Appuyer sur le boutton vert *clone and download* (pour copier le lien du repo) 

2 - Cloner le repository (dans votre dossier Ateliers) avec la commande suivante : 
```
git clone https://github.com/TheHackademy/DataGramxAirBnB.git
```

### Lancer Jupyter

Pour lancer jupyter, assurez vous de bien être positioné dans votre repo de travail (*Atelier1*) et lancer la commande suivante :  

```
jupyter notebook 
```
