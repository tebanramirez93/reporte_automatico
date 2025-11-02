# 🧠 Auditoría Inteligente de Datos  
### Análisis exploratorio, detección de patrones y exporte automatizado

Creado por: 

Jordan Esteban Ramirez Mejia
Juan Felipe Pinzon Trejo 
David Gonzalez Idarraga

Este aplicativo realiza un **análisis inteligente de datos** integrando modelos de *Machine Learning* y *Data Science* como:
- **Prophet** para detectar patrones temporales.
- **XGBoost / LightGBM** para clasificación automática.
- **KMeans / DBSCAN** para agrupación de comportamientos.
- **Isolation Forest** para detección de anomalías.
- Auditoría de **sesgos éticos y equidad** por variables sensibles.

El sistema permite:
- Cargar un archivo `.csv` con tus datos.  
- Ejecutar automáticamente un **pipeline de análisis y visualización interactiva**.  
- **Exportar un reporte en PDF** con todas las gráficas, métricas y hallazgos clave.

---

### 📊 Fuente de los datos

Los datos utilizados provienen del dataset público  
**[Lifestyle and Health Risk Prediction Synthetic Dataset](https://www.kaggle.com/)** disponible en *Kaggle*.  
Este conjunto de datos sintético fue creado con fines educativos y de demostración,  
permitiendo evaluar técnicas de análisis exploratorio, clasificación y detección de anomalías en salud.

---

### 💡 Funcionalidad destacada

> 🖨️ El aplicativo incluye una opción para **exportar automáticamente los resultados a PDF**,  
> integrando las imágenes generadas en el análisis (gráficos de correlación, distribuciones, clustering, fairness y más).  
> Esto facilita compartir reportes completos con equipos de negocio, analítica o auditoría.

---

### 🚀 Requisitos mínimos

- Python 3.9+
- Librerías: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`, `prophet`, `xgboost`,  
  `lightgbm`, `catboost`, `reportlab`, `pillow`, `kaleido`

---

**© 2025 – Proyecto académico de auditoría inteligente de datos.**  
Creado para fines educativos y demostrativos en análisis automatizado con *Streamlit + IA*.
