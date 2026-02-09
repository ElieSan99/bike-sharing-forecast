# Note Méthodologique : Prédiction de la demande Capital Bikeshare (POV)

## 1. Définition du Cas d'Usage
**Problématique** : Prédire la demande horaire de vélos sur un horizon de **48 heures (J+2)**.

**Données** : Les données brutes de trajets individuels (Capital Bikeshare 2024-2025) ont été **agrégées par pas de temps horaire** pour constituer une série temporelle de demande globale.

**Justification** : Un horizon de 48h permet à l'opérateur de planifier la logistique de rééquilibrage des stations et la maintenance préventive avant les pics de demande prévisibles (ex: rush hours en semaine).


## 2. Stratégie de Validation
- **Split Temporel Strict (3-way)** : 
  - **Train** : Jusqu'au 30/06/2025 (~75%). *Justification* : Inclut une année complète (2024) + un cycle printemps/été 2025 pour une base d'apprentissage stable.
  - **Validation** : 01/07/2025 au 31/10/2025 (~17%). *Justification* : Couvre le pic estival et la transition vers l'automne. Idéal pour optimiser les hyperparamètres sur des volumes élevés.
  - **Test** : À partir du 01/11/2025 (~8%). *Justification* : Évaluation "out-of-sample" sur la période la plus récente (fin d'année) pour tester la robustesse lors de la chute de température hivernale.

- **Absence de Data Leakage** : Utilisation uniquement de variables connues à l'instant T (temporelles) ou décalées (Lags) pour garantir la faisabilité en temps réel.

## 3. Indicateurs de Performance
L'évaluation est multidimensionnelle pour refléter les enjeux métier :
- **MAE (Mean Absolute Error)** : Indique l'erreur moyenne en nombre de vélos. Facilement interprétable par les équipes opérationnelles.
- **RMSE (Root Mean Square Error)** : Pénalise fortement les erreurs importantes, utile pour identifier les jours de "crise" logistique.
- **sMAPE (Symmetric MAPE)** : Évalue l'erreur relative (%) sans biais envers les faibles valeurs de demande (contrairement au MAPE classique).

## 4. Comparaison Baseline vs Amélioration
- **Baseline (Seasonal Naive)** : Modèle de référence simple qui prédit la demande à l'instant T comme étant égale à la demande à T-168h (même heure, même jour de la semaine précédente). Cela capture l'essentiel de la saisonnalité hebdomadaire.
- **Amélioration (LightGBM)** : L'utilisation de features multi-temporelles et de plusieurs lags permet de capturer non seulement la saisonnalité mais aussi les tendances récentes et les interactions complexes entre variables.
- **Exploration** : Une phase d'analyse exploratoire rapide (voir `notebooks/exploration.ipynb`) a permis de valider un lien entre les pics de demande et les variables temporelles (heure, jour de la semaine).


## 5. Limites et Robustesse
Les cas d'échec ont été identifiés par l'analyse des **résidus extrêmes (> 1000 vélos)** sur le set de test (Nov-Déc 2025).

Les dix plus fortes erreurs observées sont listées dans le fichier results/worst_errors.csv.

- **Cas d'échec avérés** : 
  - **Thanksgiving (27/11/2025)** : Erreur de **1028 vélos** à 17h. Le modèle prédit un pic de rush hour alors que la demande est quasi nulle.
  - **Veille de Thanksgiving (25/11/2025)** : Erreur de **1166 vélos** à 17h. Effondrement anticipé de la demande (départs en weekend) non capturé par les lags hebdomadaires.
  - **Veteran's Day (11/11/2025)** : Sur-estimation systématique (+600 vélos) lors des créneaux de pointe du matin.
- **Analyse de Robustesse** : Le modèle est performant en "régime de croisière" mais "aveugle" aux ruptures calendaires et météo. L'intégration d'un calendrier des jours fériés est la priorité n°1 pour l'industrialisation.



## 6. Vision MLOps / Passage en Run (Approche Pragmatique)
L’objectif de ce POV est de démontrer la valeur métier du modèle avant toute industrialisation lourde. Le passage en production peut se faire de manière progressive.

- **Service d’inférence** : Le modèle est encapsulé dans un service léger (ex. API REST via **FastAPI**) exposant une prédiction à J+2 à partir des données horaires récentes.
- **Chaîne de données** : Les données horaires sont mises à jour quotidiennement. Le feature engineering repose uniquement sur des informations disponibles dans le passé (lags, calendriers), garantissant la faisabilité en temps réel.
- **Monitoring de la performance** : Un suivi régulier des métriques clés (MAE, sMAPE, erreurs sur pics) est mis en place afin de détecter une dégradation des performances, notamment lors de changements d’usage ou de saisonnalité.
- **Ré-entraînement** : Le modèle peut être ré-entraîné de manière planifiée (ex. hebdomadaire) ou déclenchée suite à une baisse de performance observée. Cette étape peut être automatisée via un pipeline simple (**cron / GitHub Actions**), avant d’envisager une orchestration plus avancée.
- **Traçabilité des modèles** : Les versions du modèle, les métriques associées et les paramètres sont historisés (ex: **MLflow**) afin de garantir la reproductibilité et faciliter les comparaisons entre itérations.

Cette approche privilégie une industrialisation progressive, adaptée à un contexte réel, tout en conservant la possibilité d’évoluer vers une stack MLOps plus complète si le cas d’usage est validé.

## 7. Stratégie de Mode Dégradé (Fallback)
En production, si le modèle LightGBM échoue (ex: erreur de feature engineering sur les lags), une stratégie de fallback est prévue :
1. **Priorité 1** : Bascule sur le modèle **Baseline Seasonal Naive** (prédiction T = T-168h), extrêmement stable.
2. **Priorité 2** : Si échec critique, utilisation de la **moyenne historique** (Historical Average) par slot horaire.


