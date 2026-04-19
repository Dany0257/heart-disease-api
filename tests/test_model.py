"""
Tests de validation du modèle ML (approche MLOps).
Vérifie l'intégrité du modèle, ses performances et la cohérence des prédictions.
"""

import numpy as np
from sklearn.metrics import accuracy_score

# ─── Seuil de régression MLOps ──────────────────────────────────────────────
# Si l'accuracy tombe sous ce seuil, le pipeline CI bloque le déploiement.
ACCURACY_THRESHOLD = 0.80


class TestModelLoading:
    """Tests de chargement et d'intégrité des artefacts ML."""

    def test_model_loads_successfully(self, model):
        """Le modèle .pkl doit se charger sans erreur."""
        assert model is not None

    def test_scaler_loads_successfully(self, scaler):
        """Le scaler .pkl doit se charger sans erreur."""
        assert scaler is not None

    def test_scaler_has_correct_features(self, scaler):
        """Le scaler doit être calibré sur 10 features (variables médicales)."""
        assert scaler.n_features_in_ == 10

    def test_model_has_correct_features(self, model):
        """Le modèle doit attendre 10 features en entrée."""
        assert model.n_features_in_ == 10


class TestModelPredictions:
    """Tests de cohérence des prédictions."""

    def test_predictions_are_binary(self, model, scaler, test_dataset):
        """Les prédictions doivent être exclusivement 0 ou 1."""
        X_test, _ = test_dataset
        X_scaled = scaler.transform(X_test)
        predictions = model.predict(X_scaled)
        unique_values = set(predictions)
        assert unique_values.issubset({0, 1}), \
            f"Prédictions non-binaires détectées : {unique_values}"

    def test_probabilities_in_valid_range(self, model, scaler, test_dataset):
        """Les probabilités prédites doivent être dans [0, 1]."""
        X_test, _ = test_dataset
        X_scaled = scaler.transform(X_test)
        probabilities = model.predict_proba(X_scaled)
        assert np.all(probabilities >= 0), "Probabilités négatives détectées"
        assert np.all(probabilities <= 1), "Probabilités > 1 détectées"

    def test_probabilities_sum_to_one(self, model, scaler, test_dataset):
        """Pour chaque patient, les probabilités des classes doivent sommer à 1."""
        X_test, _ = test_dataset
        X_scaled = scaler.transform(X_test)
        probabilities = model.predict_proba(X_scaled)
        sums = probabilities.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6,
                                   err_msg="Les probabilités ne somment pas à 1")


class TestModelPerformance:
    """
    Tests de régression de performance (MLOps).
    Si le modèle est ré-entraîné avec de mauvais résultats,
    ces tests bloqueront le pipeline CI/CD.
    """

    def test_accuracy_above_threshold(self, model, scaler, test_dataset):
        """
        L'accuracy sur le test set doit rester >= 80%.
        C'est le garde-fou principal contre la régression de performance.
        """
        X_test, y_test = test_dataset
        X_scaled = scaler.transform(X_test)
        predictions = model.predict(X_scaled)
        accuracy = accuracy_score(y_test, predictions)
        assert accuracy >= ACCURACY_THRESHOLD, \
            f"RÉGRESSION DÉTECTÉE : accuracy={accuracy:.2%} < seuil={ACCURACY_THRESHOLD:.0%}"

    def test_model_not_always_same_class(self, model, scaler, test_dataset):
        """
        Le modèle ne doit pas prédire toujours la même classe
        (signe d'un modèle dégénéré ou mal entraîné).
        """
        X_test, _ = test_dataset
        X_scaled = scaler.transform(X_test)
        predictions = model.predict(X_scaled)
        unique_predictions = set(predictions)
        assert len(unique_predictions) > 1, \
            "Le modèle prédit toujours la même classe — modèle dégénéré !"
