# app.py
# Ejecuta: streamlit run app.py
import warnings
warnings.filterwarnings("ignore")

import os, io
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st

# Visualizaciones
import plotly.express as px
import plotly.graph_objects as go

# ML / Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Modelos opcionales (fallback a RF si no están)
HAS_XGB = HAS_LGBM = HAS_CAT = False
try:
    from xgboost import XGBClassifier; HAS_XGB = True
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier; HAS_LGBM = True
except Exception:
    pass
try:
    from catboost import CatBoostClassifier; HAS_CAT = True
except Exception:
    pass

# Prophet opcional
HAS_PROPHET = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    try:
        from fbprophet import Prophet
        HAS_PROPHET = True
    except Exception:
        HAS_PROPHET = False

# Utilidades
def preprocess_basic(df: pd.DataFrame):
    """Rellena nulos y normaliza categóricas (str.strip/Unknown)."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()

    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        s = df[c].astype(str).str.strip()
        s = s.mask(s.eq("") | s.eq("nan"), "Unknown")
        df[c] = s.fillna("Unknown")
    return df, num_cols, cat_cols

def one_hot_limited(df: pd.DataFrame, max_unique: int = 60):
    """One-hot encode de categóricas con cardinalidad limitada y saneo final."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    to_dummy = [c for c in cat_cols if df[c].nunique() <= max_unique]
    df = pd.get_dummies(df, columns=to_dummy, dummy_na=False)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def train_classifier(X, y):
    """Entrena CatBoost/LGBM/XGBoost → fallback RF. Devuelve métricas y artefactos."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_name = "RandomForest"
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

    if HAS_CAT:
        try:
            model_name = "CatBoost"
            model = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1, verbose=False)
        except Exception:
            model_name = "RandomForest"; model = RandomForestClassifier(n_estimators=300, random_state=42)
    elif HAS_LGBM:
        try:
            model_name = "LightGBM"
            model = LGBMClassifier(n_estimators=500, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
        except Exception:
            model_name = "RandomForest"; model = RandomForestClassifier(n_estimators=300, random_state=42)
    elif HAS_XGB:
        try:
            model_name = "XGBoost"
            model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, random_state=42, eval_metric="logloss", tree_method="hist")
        except Exception:
            model_name = "RandomForest"; model = RandomForestClassifier(n_estimators=300, random_state=42)

    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        yprob = model.predict_proba(Xte)[:, 1]
        ypred = (yprob >= 0.5).astype(int)
    else:
        yprob = model.predict(Xte)
        ypred = yprob

    try:
        auc = roc_auc_score(yte, yprob)
    except Exception:
        auc = float("nan")

    conf_mat = confusion_matrix(yte, ypred)
    report = classification_report(yte, ypred, zero_division=0, output_dict=True)
    return model_name, model, (Xtr, Xte, ytr, yte), auc, conf_mat, report

def clusterize(df_num: pd.DataFrame, k=5, eps=1.5, min_samples=20):
    Z = StandardScaler().fit_transform(df_num.values)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_km = km.fit_predict(Z)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels_db = db.fit_predict(Z)

    pca = PCA(n_components=2, random_state=42)
    comp = pca.fit_transform(Z)
    return labels_km, labels_db, comp

def detect_anomalies(df_num: pd.DataFrame, contamination=0.03):
    Z = StandardScaler().fit_transform(df_num.values)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=42)
    lab = iso.fit_predict(Z)
    scores = -iso.score_samples(Z)
    return lab, scores

def fairness_audit(df: pd.DataFrame, sensitive_col: str, target_col: str):
    tmp = df[[sensitive_col, target_col]].dropna().copy()
    tmp[target_col] = tmp[target_col].astype(int)
    grp = tmp.groupby(sensitive_col)[target_col].agg(['mean','count']).rename(columns={'mean':'positive_rate','count':'n'})
    ref_group, ref_rate = None, None
    if not grp.empty and grp['positive_rate'].min() > 0:
        ref_group = grp['positive_rate'].idxmin()
        ref_rate  = grp.loc[ref_group, 'positive_rate']
        grp['disparate_impact'] = grp['positive_rate'] / ref_rate
    return grp, ref_group, ref_rate

def build_synthetic_daily(df: pd.DataFrame, metric_col: str, days: int = 120):
    rng = pd.date_range(end=date.today(), periods=days, freq="D")
    tmp = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    tmp["ds"] = np.random.choice(rng, size=len(tmp))
    daily = tmp.groupby("ds")[metric_col].mean().reset_index().rename(columns={metric_col:"y"})
    return daily

# Imagenes para reporte en PDF
# Requiere: reportlab, pillow, kaleido
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4 as RL_A4   # alias seguro
from reportlab.lib.units import cm as RL_CM       # alias seguro
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

os.makedirs("tmp_imgs", exist_ok=True)

def save_plotly_as_image(fig, filename: str) -> str:
    """Guarda figura Plotly como PNG (requiere 'kaleido') y valida con Pillow."""
    path = os.path.join("tmp_imgs", filename)
    fig.write_image(path)
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with PILImage.open(path) as im:
                im.verify()
        else:
            raise ValueError("Imagen no creada o vacía.")
    except Exception as e:
        try:
            os.remove(path)
        except Exception:
            pass
        raise e
    return path

def _valid_pngs(paths):
    """Filtra PNGs válidos (abren con Pillow y pesan > 0)."""
    out = []
    for p in (paths or []):
        try:
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                continue
            with PILImage.open(p) as im:
                im.verify()
            out.append(p)
        except Exception:
            continue
    return out

def build_pdf_report(title: str, hallazgos: list, image_paths: list) -> bytes:
    """Genera PDF con título, bullets e imágenes escaladas por ancho (robusto a choques de nombres)."""
    hallazgos = [str(h) for h in (hallazgos or []) if h is not None]
    image_paths = _valid_pngs(image_paths)

    buffer = io.BytesIO()
    # Usar RL_A4 y RL_CM (todo en floats)
    page_w, page_h = tuple(map(float, RL_A4))
    left_margin = 1.5 * float(RL_CM)
    right_margin = 1.5 * float(RL_CM)
    top_margin = 1.5 * float(RL_CM)
    bottom_margin = 1.5 * float(RL_CM)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=(page_w, page_h),
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBig", parent=styles["Title"], fontSize=20, leading=24, spaceAfter=12))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#0f4c81"), spaceAfter=6))
    styles.add(ParagraphStyle(name="NormalSmall", parent=styles["BodyText"], fontSize=10, leading=13))

    story = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Título
    story.append(Paragraph(str(title or "Reporte"), styles["TitleBig"]))
    story.append(Paragraph(f"Generado: {now_str}", styles["NormalSmall"]))
    story.append(Spacer(1, 10))

    # Hallazgos
    story.append(Paragraph("Hallazgos clave", styles["Section"]))
    if hallazgos:
        items = [ListItem(Paragraph(h, styles["BodyText"]), bulletColor=colors.black) for h in hallazgos]
        story.append(ListFlowable(items, bulletType="bullet", leftIndent=10))
    else:
        story.append(Paragraph("Sin hallazgos automáticos destacados.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Imágenes
    story.append(Paragraph("Gráficas del análisis", styles["Section"]))
    max_width = page_w - (left_margin + right_margin)  # float puro

    for p in image_paths:
        try:
            # Validar tamaño con Pillow (asegura escalares)
            with PILImage.open(p) as im:
                w, h = im.size
                float(w); float(h)

            # Escalar por ANCHO únicamente (altura proporcional)
            rl_img = RLImage(p, width=max_width)
            story.append(rl_img)
            story.append(Spacer(1, 8))
            story.append(Paragraph(os.path.basename(p), styles["NormalSmall"]))
            story.append(Spacer(1, 12))
        except Exception as e:
            story.append(Paragraph(f"No se pudo insertar imagen: {os.path.basename(p)} ({e})", styles["NormalSmall"]))
            story.append(Spacer(1, 6))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# UI
st.set_page_config(page_title="Auditoría Inteligente", layout="wide")
st.title("🧠 Auditoría Inteligente – E2E")
st.markdown("Creado por Juan Felipe Pinzon Trejo, David Gonzalez Idarraga y Jordan Esteban Ramirez Mejia")

with st.sidebar:
    st.header("⚙️ Configuración")
    file = st.file_uploader("Cargar CSV", type=["csv"])
    # Ruta por defecto que pediste:
    DEFAULT_DEMO_PATH = "dataset/Lifestyle_and_Health_Risk_Prediction_Synthetic_Dataset.csv"
    demo = st.checkbox("Usar dataset de demo (salud)", value=True if not file else False)
    metric_for_prophet = st.selectbox("Métrica para Prophet (serie sintética)", options=["bmi","sleep","weight","height"], index=0)
    k_kmeans = st.slider("KMeans: k clusters", min_value=2, max_value=10, value=5, step=1)
    eps_db = st.slider("DBSCAN: eps", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    min_s_db = st.slider("DBSCAN: min_samples", min_value=5, max_value=200, value=20, step=5)
    contamination = st.slider("IsolationForest: contamination", min_value=0.01, max_value=0.10, value=0.03, step=0.01)

# Carga del archivo
if file:
    df = pd.read_csv(file)
elif demo:
    # ← usa tu ruta pedida
    df = pd.read_csv(DEFAULT_DEMO_PATH)
else:
    st.info("Carga un CSV o marca 'Usar dataset de demo'.")
    st.stop()

st.subheader("👀 Vista previa")
st.write(df.head())
st.caption(f"Filas: {len(df):,} | Columnas: {len(df.columns)}")
st.write("Tipos de datos:", dict(df.dtypes.astype(str).value_counts()))

# Pre-Procesamiento
st.header("1) Preprocesamiento automático")
df_pp, num_cols, cat_cols = preprocess_basic(df)
st.write("Numéricas:", num_cols)
st.write("Categóricas:", cat_cols)
st.success("Nulos procesados, categóricas normalizadas.")

# EDA
st.header("2) EDA (Exploratorio)")
img_paths = []  # rutas de imágenes para PDF

col1, col2 = st.columns(2)
with col1:
    st.subheader("Estadísticas numéricas")
    if num_cols:
        st.dataframe(df_pp[num_cols].describe().T)
    else:
        st.warning("No hay columnas numéricas.")

with col2:
    if len(num_cols) >= 2:
        st.subheader("Correlación")
        corr = df_pp[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues", title="Matriz de correlación")
        st.plotly_chart(fig_corr, use_container_width=True)
        try:
            img_paths.append(save_plotly_as_image(fig_corr, "01_correlacion.png"))
        except Exception as e:
            st.warning(f"No se pudo guardar imagen de correlación: {e}")

st.subheader("Distribuciones")
if num_cols:
    for c in num_cols:
        fig_hist = px.histogram(df_pp, x=c, nbins=30, marginal="box", title=f"Distribución: {c}")
        st.plotly_chart(fig_hist, use_container_width=True)
        try:
            img_paths.append(save_plotly_as_image(fig_hist, f"hist_{c}.png"))
        except Exception as e:
            st.warning(f"No se pudo guardar histograma de {c}: {e}")

# Clasificación
st.header("3) Clasificación (High Risk vs otros)")
model_name = None
auc = float("nan")
if "health_risk" in df_pp.columns:
    df_model = df_pp.copy()
    df_model["target"] = df_model["health_risk"].str.contains("High", case=False, na=False).astype(int)

    X_raw = df_model.drop(columns=["target"])
    drop_guess = [c for c in X_raw.columns if any(k in c.lower() for k in ["id","uuid","guid","date","time","timestamp"])]
    X_raw = X_raw.drop(columns=drop_guess, errors="ignore")
    X = one_hot_limited(X_raw, max_unique=60)
    y = df_model["target"].values

    if X.shape[1] >= 2 and y.sum() > 0 and y.sum() < len(y):
        model_name, model, splits, auc, conf_mat, report = train_classifier(X, y)
        Xtr, Xte, ytr, yte = splits

        st.write(f"**Modelo:** {model_name}")
        st.write(f"**ROC-AUC:** {auc:.3f}" if np.isfinite(auc) else "ROC-AUC: N/A")

        fig_cm = px.imshow(conf_mat, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Predicho", y="Real", color=""), title="Matriz de confusión")
        st.plotly_chart(fig_cm, use_container_width=True)
        try:
            img_paths.append(save_plotly_as_image(fig_cm, "02_matriz_confusion.png"))
        except Exception as e:
            st.warning(f"No se pudo guardar imagen de matriz de confusión: {e}")

        rep_df = pd.DataFrame(report).T
        st.dataframe(rep_df.round(3))

        if hasattr(model, "feature_importances_"):
            imps = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(25)
            fig_imp = px.bar(imps[::-1], orientation="h", title="Top 25 importancias")
            st.plotly_chart(fig_imp, use_container_width=True)
            try:
                img_paths.append(save_plotly_as_image(fig_imp, "03_importancias.png"))
            except Exception as e:
                st.warning(f"No se pudo guardar imagen de importancias: {e}")
    else:
        st.warning("No hay señal suficiente para clasificar (target constante o muy desbalanceado).")
else:
    st.info("No existe 'health_risk' en el dataset. Para clasificar, agrega/deriva una etiqueta binaria.")

# Clustering
st.header("4) Clustering (KMeans / DBSCAN)")
num_for_cluster = [c for c in num_cols if df_pp[c].dtype.kind in "if"]
labels_km = labels_db = None
if len(num_for_cluster) >= 2:
    df_num = df_pp[num_for_cluster]
    labels_km, labels_db, comp = clusterize(df_num, k=k_kmeans, eps=eps_db, min_samples=min_s_db)

    df_clu = pd.DataFrame({"PC1": comp[:,0], "PC2": comp[:,1],
                           "KMeans": labels_km.astype(str), "DBSCAN": labels_db.astype(str)})
    st.write("**KMeans – Distribución:**")
    st.dataframe(pd.Series(labels_km).value_counts().sort_index().rename("count"))

    st.write("**DBSCAN – Distribución (-1 = ruido):**")
    st.dataframe(pd.Series(labels_db).value_counts().sort_index().rename("count"))

    fig_km = px.scatter(df_clu, x="PC1", y="PC2", color="KMeans", title="PCA 2D (KMeans)")
    st.plotly_chart(fig_km, use_container_width=True)
    try:
        img_paths.append(save_plotly_as_image(fig_km, "04_kmeans_pca.png"))
    except Exception as e:
        st.warning(f"No se pudo guardar imagen KMeans PCA: {e}")

    fig_db = px.scatter(df_clu, x="PC1", y="PC2", color="DBSCAN", title="PCA 2D (DBSCAN)")
    st.plotly_chart(fig_db, use_container_width=True)
    try:
        img_paths.append(save_plotly_as_image(fig_db, "05_dbscan_pca.png"))
    except Exception as e:
        st.warning(f"No se pudo guardar imagen DBSCAN PCA: {e}")
else:
    st.warning("Se requieren ≥2 columnas numéricas para clustering.")

# Anomalias
st.header("5) Detección de anomalías (IsolationForest)")
n_anom = None
if len(num_for_cluster) >= 2:
    df_num = df_pp[num_for_cluster]
    lab, scores = detect_anomalies(df_num, contamination=contamination)
    n_anom = int((lab == -1).sum())
    st.metric("Anomalías detectadas", n_anom)

    fig_scores = px.histogram(x=scores, nbins=50, title="Distribución de anomaly_score")
    st.plotly_chart(fig_scores, use_container_width=True)
    try:
        img_paths.append(save_plotly_as_image(fig_scores, "06_anomalias_scores.png"))
    except Exception as e:
        st.warning(f"No se pudo guardar imagen de anomaly_score: {e}")

    anom_table = df_pp.copy()
    anom_table["anomaly"] = (lab == -1).astype(int)
    anom_table["anomaly_score"] = scores
    st.subheader("Top anomalías")
    st.dataframe(anom_table.sort_values("anomaly_score", ascending=False).head(50))
else:
    st.warning("Se requieren ≥2 columnas numéricas para anomalías.")

# Fairness
st.header("6) Auditoría de sesgo (Fairness)")
fair = pd.DataFrame()
cand_sens = [c for c in df_pp.columns if df_pp[c].dtype == "object" and df_pp[c].nunique() <= 30]
sensitive_col = st.selectbox("Variable sensible", options=cand_sens or ["(no disponible)"])
if "health_risk" in df_pp.columns:
    df_fair = df_pp.copy()
    df_fair["target"] = df_fair["health_risk"].str.contains("High", case=False, na=False).astype(int)
    if sensitive_col != "(no disponible)":
        fair, ref_group, ref_rate = fairness_audit(df_fair, sensitive_col, "target")
        if not fair.empty:
            fair_sorted = fair.sort_values("positive_rate", ascending=False)
            st.dataframe(fair_sorted)
            try:
                fig_fair = px.bar(fair_sorted.reset_index(), x="positive_rate", y=sensitive_col,
                                  orientation="h", title=f"Fairness – tasa positiva por {sensitive_col}")
                st.plotly_chart(fig_fair, use_container_width=True)
                img_paths.append(save_plotly_as_image(fig_fair, "07_fairness.png"))
            except Exception as e:
                st.warning(f"No se pudo guardar imagen de fairness: {e}")

            if "disparate_impact" in fair.columns:
                st.info(f"Grupo de referencia: **{ref_group}** (tasa={ref_rate:.3f}). DI ideal≈1; >1.25 puede sugerir sesgo.")
        else:
            st.warning("No hay datos suficientes para calcular fairness.")
    else:
        st.warning("No se encontró variable sensible adecuada (≤30 categorías).")
else:
    st.info("Fairness requiere derivar 'target' a partir de 'health_risk'.")

# Prophet sintético
st.header("7) Patrones temporales (Prophet – serie sintética)")
daily = pd.DataFrame(); forecast = pd.DataFrame()
if HAS_PROPHET:
    if metric_for_prophet in df_pp.columns and df_pp[metric_for_prophet].dtype.kind in "if":
        daily = build_synthetic_daily(df_pp, metric_for_prophet, days=120)
        if len(daily) >= 10:
            m = Prophet(seasonality_mode="additive")
            m.fit(daily)
            future = m.make_future_dataframe(periods=14)
            forecast = m.predict(future)

            fig_ts_hist = px.line(daily, x="ds", y="y", title=f"{metric_for_prophet} promedio diario (histórico sintético)")
            st.plotly_chart(fig_ts_hist, use_container_width=True)
            try:
                img_paths.append(save_plotly_as_image(fig_ts_hist, "08_ts_historico.png"))
            except Exception as e:
                st.warning(f"No se pudo guardar imagen de histórico Prophet: {e}")

            fig_fore = go.Figure()
            fig_fore.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="yhat"))
            fig_fore.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="yhat_upper", line=dict(dash="dash")))
            fig_fore.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="yhat_lower", line=dict(dash="dash")))
            fig_fore.update_layout(title=f"Pronóstico {metric_for_prophet} (+14 días)")
            st.plotly_chart(fig_fore, use_container_width=True)
            try:
                img_paths.append(save_plotly_as_image(fig_fore, "09_ts_forecast.png"))
            except Exception as e:
                st.warning(f"No se pudo guardar imagen de forecast Prophet: {e}")
        else:
            st.warning("No hay suficientes puntos diarios para Prophet (>=10).")
    else:
        st.warning("Selecciona una métrica numérica válida para Prophet.")
else:
    st.info("Prophet no está instalado. Instala con: `pip install prophet`.")

# Funcionalidad para reporte PDF
st.header("8) Reporte automático (PDF con imágenes)")
hallazgos = []
try:
    if 'labels_km' in locals() and labels_km is not None:
        n_km = pd.Series(labels_km).nunique()
        hallazgos.append(f"Se detectaron {n_km} clusters (KMeans).")
except Exception:
    pass
try:
    if 'labels_db' in locals() and labels_db is not None:
        n_db = pd.Series(labels_db).nunique()
        ruido = int((pd.Series(labels_db) == -1).sum())
        hallazgos.append(f"DBSCAN generó {n_db} grupos (ruido={ruido}).")
except Exception:
    pass
try:
    if np.isfinite(auc):
        hallazgos.append(f"Clasificador {model_name} con ROC-AUC={auc:.3f}.")
except Exception:
    pass
try:
    if 'n_anom' in locals() and n_anom is not None:
        hallazgos.append(f"Anomalías detectadas: {n_anom} (IsolationForest).")
except Exception:
    pass
try:
    if HAS_PROPHET and not daily.empty and not forecast.empty:
        last_obs = daily.sort_values("ds").iloc[-1]["y"]
        last_fore = forecast.sort_values("ds").iloc[-1]["yhat"]
        delta = last_fore - last_obs
        tendencia = "al alza" if delta > 0 else "a la baja" if delta < 0 else "estable"
        hallazgos.append(f"Prophet: {metric_for_prophet} proyecta tendencia {tendencia} (Δ {delta:.2f}).")
except Exception:
    pass

# Recoger imágenes válidas
img_paths = sorted([os.path.join("tmp_imgs", f) for f in os.listdir("tmp_imgs") if f.lower().endswith(".png")])
img_paths = _valid_pngs(img_paths)

if st.button("🖨️ Generar PDF con imágenes"):
    if not img_paths and not hallazgos:
        st.warning("No hay imágenes ni hallazgos para incluir en el PDF todavía.")
    else:
        pdf_bytes = build_pdf_report(
            title="Resumen de Auditoría Inteligente",
            hallazgos=hallazgos,
            image_paths=img_paths
        )
        st.success("PDF generado.")
        st.download_button(
            label="⬇️ Descargar PDF",
            data=pdf_bytes,
            file_name="reporte_auditoria.pdf",
            mime="application/pdf"
        )
        st.info("Si falta alguna gráfica, asegúrate de guardarla con save_plotly_as_image en su bloque.")
