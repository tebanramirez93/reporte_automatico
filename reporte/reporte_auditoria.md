# Resumen de Auditoría Inteligente

- Se detectaron 5 clusters (KMeans).
- DBSCAN generó 1 grupos (ruido=0).
- Se detectaron 150 registros anómalos (IsolationForest).
- Clasificador CatBoost con ROC-AUC=1.000.
- Fairness: 'low' presenta mayor disparate impact vs. grupo de referencia.
- Prophet: bmi proyecta tendencia a la baja (Δ -0.41).