import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Charger le dataset
df = pd.read_csv("data.csv")

#Supprimer les espaces dans les noms des colonnes
df.columns = df.columns.str.strip()

#Remplacer les valeurs manquantes par NaN
df = df.replace('?', np.nan)

#Si une colonne a trop de valeurs manquantes (>50%), 
# on la supprime elle n'apporterait que du bruit au modèle.
# Si peu de valeurs manquent, on les impute (remplace) par la médiane 
# (plus robuste que la moyenne car insensible aux valeurs extrêmes)

# On supprime les colonnes ca, thal et slope car elles ont trop de valeurs manquantes
df = df.drop(columns=["ca", "thal", "slope"])

# On convertit les colonnes en numérique
cols_to_convert = ["trestbps", "chol", "fbs", "restecg", "thalach", "exang"]
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric)

# On remplace les valeurs manquantes par la médiane
df = df.fillna(df.median())

# Séparer les variables explicatives (X) et la cible (y)
x = df.drop(columns=["num"])
y = df["num"]

print("Shape de X :", x.shape)
print("Shape de y :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

print("Taille train :", X_train.shape)
print("Taille test  :", X_test.shape)

"""
Regarde bien les variables dont on dispose :

L'âge va de 28 à 66 ans.
Le cholestérol va de 100 à près de 400 mg/dl.
Le sexe, lui, vaut 0 ou 1.
Pour un algorithme de machine learning 
(comme la Régression Logistique qu'on va utiliser), 
des variables avec des échelles très différentes, 
c'est un problème. Il risque de croire que le cholestérol est 
"plus important" que le sexe juste parce que les nombres sont plus gros !

On va donc "normaliser" (ou standardiser) les données. 
Ça veut dire qu'on va tout ramener à la même échelle 
(autour de 0, avec un écart-type de 1).
"""
# Créer le "normalisateur"
scaler = StandardScaler()

# On "apprend" l'échelle sur le train et on l'applique (fit_transform)
X_train_scaled = scaler.fit_transform(X_train)

# On applique la MÊME échelle sur le test (transform) - on n'apprend pas dessus !
X_test_scaled = scaler.transform(X_test)

# Créer le modèle de Régression Logistique
model = LogisticRegression(random_state=42)

# 2. Entraîner le modèle sur les données d'entraînement (normalisées)
model.fit(X_train_scaled, y_train)

# 3. Faire des prédictions sur les données de test (jamais vues par le modèle)
y_pred = model.predict(X_test_scaled)

# 4. Évaluer les performances
print("\n--- Évaluation du Modèle (Régression Logistique) ---")
print("Accuracy  :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall    :", recall_score(y_test, y_pred))
print("F1-Score  :", f1_score(y_test, y_pred))
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))


# Créer le modèle Random Forest
# n_estimators=100 veut dire qu'on crée 100 arbres de décision
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
rf_model.fit(X_train_scaled, y_train)

# Faire les prédictions
rf_y_pred = rf_model.predict(X_test_scaled)

# Évaluer les performances du Random Forest
print("\n--- Évaluation du Modèle (Random Forest) ---")
print("Accuracy  :", accuracy_score(y_test, rf_y_pred))
print("Precision :", precision_score(y_test, rf_y_pred))
print("Recall    :", recall_score(y_test, rf_y_pred))
print("F1-Score  :", f1_score(y_test, rf_y_pred))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, rf_y_pred))


# Identifier les variables les plus importantes
print("\n--- Importance des Variables (Feature Importance) ---")
feature_importances = pd.DataFrame(
rf_model.feature_importances_,index = x.columns,columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)

# 6. Sauvegarder le Modèle et le Scaler
joblib.dump(rf_model, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n Modèle et Scaler sauvegardés avec succès ! ")
