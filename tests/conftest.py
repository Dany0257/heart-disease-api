"""
Fixtures pytest partagées pour le projet Heart Disease API.
Chaque fixture est chargée une seule fois par session de tests (scope="session").
"""

import pytest
import joblib
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from api import app


# ─── Fixtures API ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """Client HTTP de test pour interroger l'API FastAPI."""
    return TestClient(app)


# ─── Fixtures Modèle ML ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model():
    """Charge le modèle Random Forest sérialisé."""
    return joblib.load("heart_disease_model.pkl")


@pytest.fixture(scope="session")
def scaler():
    """Charge le StandardScaler sérialisé."""
    return joblib.load("scaler.pkl")


@pytest.fixture(scope="session")
def test_dataset():
    """
    Recrée le test set à partir du CSV original (même random_state=42)
    pour valider les performances du modèle de manière reproductible.
    """
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    df = df.replace('?', np.nan)
    df = df.drop(columns=["ca", "thal", "slope"])

    cols_to_convert = ["trestbps", "chol", "fbs", "restecg", "thalach", "exang"]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric)
    df = df.fillna(df.median())

    X = df.drop(columns=["num"])
    y = df["num"]

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_test, y_test


# ─── Profils patients de test ────────────────────────────────────────────────

@pytest.fixture
def healthy_patient():
    """Profil clinique d'un patient probablement sain."""
    return {
        "age": 35,
        "sex": 0,
        "cp": 1,
        "trestbps": 120.0,
        "chol": 180.0,
        "fbs": 0.0,
        "restecg": 0.0,
        "thalach": 170.0,
        "exang": 0.0,
        "oldpeak": 0.0
    }


@pytest.fixture
def sick_patient():
    """Profil clinique d'un patient à haut risque cardiaque."""
    return {
        "age": 65,
        "sex": 1,
        "cp": 4,
        "trestbps": 160.0,
        "chol": 300.0,
        "fbs": 1.0,
        "restecg": 2.0,
        "thalach": 100.0,
        "exang": 1.0,
        "oldpeak": 4.0
    }
