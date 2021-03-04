# Architecture étudiée
Pour les features extraction :

```
  Import CSV
  Normalisé les images et les landmarks
  Extraire les features :
    Geometrie en utilisant uniquement les landmarks et leurs distances relative
    Texture en utilisant le gradient, contrast, eyc
  Générer un CSV contenant les features
```
