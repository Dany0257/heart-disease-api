"""
Tests fonctionnels de l'API FastAPI Heart Disease.
Vérifie les endpoints, la validation des données et le format des réponses.
"""


class TestHomeEndpoint:
    """Tests pour l'endpoint GET /"""

    def test_home_status_code(self, client):
        """L'endpoint d'accueil doit retourner un code 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_home_message(self, client):
        """L'endpoint d'accueil doit retourner le message de bienvenue."""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "prête" in data["message"] or "API" in data["message"]


class TestPredictEndpoint:
    """Tests pour l'endpoint POST /predict"""

    def test_predict_returns_200(self, client, healthy_patient):
        """Une requête valide doit retourner un code 200."""
        response = client.post("/predict", json=healthy_patient)
        assert response.status_code == 200

    def test_predict_response_format(self, client, healthy_patient):
        """La réponse doit contenir les 3 clés attendues."""
        response = client.post("/predict", json=healthy_patient)
        data = response.json()
        assert "prediction" in data
        assert "resultat" in data
        assert "probabilite" in data

    def test_predict_binary_output(self, client, healthy_patient):
        """La prédiction doit être 0 ou 1."""
        response = client.post("/predict", json=healthy_patient)
        data = response.json()
        assert data["prediction"] in [0, 1]

    def test_predict_healthy_patient(self, client, healthy_patient):
        """Un profil sain doit être prédit comme sain (prediction=0)."""
        response = client.post("/predict", json=healthy_patient)
        data = response.json()
        assert data["prediction"] == 0
        assert "Sain" in data["resultat"]

    def test_predict_sick_patient(self, client, sick_patient):
        """Un profil à haut risque doit être prédit comme malade (prediction=1)."""
        response = client.post("/predict", json=sick_patient)
        data = response.json()
        assert data["prediction"] == 1
        assert "Maladie" in data["resultat"] or "détectée" in data["resultat"]

    def test_predict_probability_format(self, client, healthy_patient):
        """La probabilité doit être un string formaté avec un %."""
        response = client.post("/predict", json=healthy_patient)
        data = response.json()
        assert "%" in data["probabilite"]


class TestInputValidation:
    """Tests de validation des données d'entrée (Pydantic)."""

    def test_missing_fields_returns_422(self, client):
        """Des données incomplètes doivent retourner une erreur 422."""
        incomplete_data = {"age": 50, "sex": 1}
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        """Un corps de requête vide doit retourner une erreur 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_wrong_type_returns_422(self, client):
        """Des types incorrects doivent retourner une erreur 422."""
        bad_data = {
            "age": "trente",  # string au lieu de int
            "sex": 1, "cp": 1, "trestbps": 120.0, "chol": 180.0,
            "fbs": 0.0, "restecg": 0.0, "thalach": 170.0,
            "exang": 0.0, "oldpeak": 0.0
        }
        response = client.post("/predict", json=bad_data)
        assert response.status_code == 422
