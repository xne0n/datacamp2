# Architecture étudiée
Pour les features extraction :

```
  Import CSV
  Normalisé les images et les landmarks
  Extraire les features :
    Geometrie en utilisant uniquement les landmarks et leurs distances relative
    Texture en utilisant le gradient, contrast, etc...
  Générer un CSV contenant les features
```
# Questions pour M. Gonzalez
```
- Pour utiliser les features extraites via la texture doit on normalisé toutes les images ?
      Si oui mieux vaut-il recadrer les images pour garder uniquement les visages et accelerer le temps de calcul ?
- Peut on utiliser une Random Forest ?
```
