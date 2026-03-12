from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# 1. Créer l'application FastAPI
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API pour prédire le risque de maladie cardiaque",
    version="1.0"
)
#ça je le mets pour que mon application React puisse communiquer avec mon API pour le moment
# On définit qui a le droit d'interroger notre API
origines_autorisees = [
    "http://localhost:5173", # C'est le port par défaut qu'utilisera notre outil React (Vite) en local
    "http://localhost:3000", # Au cas où
    # On ajoutera l'URL publique de Vercel/Render plus tard ici !
]
# On active le middleware CORS pour autoriser ces origines
app.add_middleware(
    CORSMiddleware,
    allow_origins=origines_autorisees,
    allow_credentials=True,
    allow_methods=["*"], # Autorise toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"], # Autorise tous les types de headers
)
# 2. Charger le modèle et le scaler au démarrage de l'API
# On utilise r pour être sûr de bien lire le fichier
try:
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Modèles chargés avec succès !")
except Exception as e:
    print(f"Erreur de chargement des modèles : {e}")

# 3. Définir le "moule" des données attendues par l'API (avec Pydantic)
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
# 4. Route de test basique
@app.get("/")
def home():
    return {"message": "API Heart Disease prête à l'emploi !"}
# 5. Route de prédiction (le coeur de l'API !)
@app.post("/predict")
def predict_heart_disease(data: PatientData):
    # a. Convertir les données reçues en dictionnaire (Pandas a besoin d'une liste de dict)
    input_data = pd.DataFrame([data.model_dump()])
    
    # b. Normaliser les données avec le scaler (comme on a fait pour X_test)
    input_scaled = scaler.transform(input_data)
    
    # c. Demander au modèle de faire une prédiction
    prediction = model.predict(input_scaled)[0]
    
    # d. Demander au modèle son niveau de certitude (probabilité)
    probability = model.predict_proba(input_scaled)[0][prediction]
    
    # e. Préparer la réponse
    result = "Maladie détectée (Risque élevé)" if prediction == 1 else "Pas de maladie (Sain)"
    
    return {
        "prediction": int(prediction),
        "resultat": result,
        "probabilite": f"{probability * 100:.1f} %"
    }
