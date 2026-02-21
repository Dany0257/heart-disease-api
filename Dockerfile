# 1. On part d'un Linux très léger avec Python 3.10 déjà installé
FROM python:3.10-slim

# 2. On se place dans le dossier /app à l'intérieur de la boîte
WORKDIR /app

# 3. On copie seulement notre "liste de courses"
COPY requirements.txt .

# 4. On installe toutes les bibliothèques de la liste
RUN pip install --no-cache-dir -r requirements.txt

# 5. On copie tout le reste de notre projet (api.py, model.pkl, etc.) dans la boîte
COPY . .

# 6. On indique que notre boîte va communiquer sur le port 8000
EXPOSE 8000

# 7. La commande exécutée quand on "allume" la boîte (similaire à ce que tu as tapé)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
