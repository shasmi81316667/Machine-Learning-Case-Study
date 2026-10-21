---
title: Simple ML Predictor
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ML Studio (Simple ML Predictor)

One small Streamlit app with two tasks:

- **Classification**: 10 models learn to name the fault on a steel plate. Type in measurements, pick a model, get a prediction. A second tab compares all 10 models (one table with Accuracy, Precision, Recall, F1, ROC-AUC, plus grouped bars, heatmap, radar, ROC curves, confusion matrices and feature importance).
- **Clustering**: K-Means and Agglomerative (Ward) group customers into segments. Compare them (Silhouette, Davies-Bouldin, Calinski-Harabasz), see PCA maps for both algorithms, t-SNE maps, the K-Means elbow curve and the Hierarchical dendrogram, and describe a new customer to see which segment they fall in.

## Run it on your computer

1. Put `marketing_campaign.csv` (tab-separated, from your archive.zip) in the `data/` folder.
   `faults_with_missing.csv` is already there. If you skip this, the Clustering page lets you upload the file instead.
2. In a terminal inside this folder:

        python -m venv venv
        venv\Scripts\activate            (Windows)
        source venv/bin/activate         (Mac / Linux)
        pip install -r requirements.txt
        streamlit run app.py

3. Your browser opens at http://localhost:8501. The first start trains the models (about 15 seconds).

## Put it online

See **DEPLOY.md** for click-by-click steps for Streamlit Community Cloud, Hugging Face Spaces and Render.

## Files

| File | What it does |
|---|---|
| `app.py` | The screens (everything you see) |
| `ml.py` | Training, scoring and predicting |
| `data/` | The two CSV files |
| `requirements.txt` | Libraries to install |
| `Dockerfile` | Used by Hugging Face |
| `render.yaml` | Used by Render |
