import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay, classification_report
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="ML Playground: Supervisado y No Supervisado", layout="wide")

st.title("🧠 Machine Learning Playground")
st.caption("Aprendizaje **supervisado** (clasificación) con datos sintéticos y visualizaciones interactivas.")

# =========================
# Sidebar — parámetros de datos
# =========================
st.sidebar.header("⚙️ Generación de Datos Sintéticos")

n_samples = st.sidebar.slider("Número de registros (≥ 300)", min_value=300, max_value=10000, value=600, step=100)
n_features = st.sidebar.slider("Número de columnas numéricas", min_value=6, max_value=20, value=6, step=1)
n_informative = st.sidebar.slider("Características informativas", min_value=2, max_value=n_features, value=min(4, n_features), step=1)
n_redundant = st.sidebar.slider("Características redundantes", min_value=0, max_value=max(0, n_features-1), value=1, step=1)
class_sep = st.sidebar.slider("Separación entre clases", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
flip_y = st.sidebar.slider("Ruido (flip_y)", min_value=0.0, max_value=0.3, value=0.02, step=0.01)
weight_majority = st.sidebar.slider("Proporción clase 0 (balance de clases)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

random_state = st.sidebar.number_input("Random state", min_value=0, value=42)
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
standardize = st.sidebar.checkbox("Estandarizar variables numéricas", value=True)

st.sidebar.markdown("---")
st.sidebar.header("🤖 Modelos a entrenar")
default_models = ["GaussianNB", "LogisticRegression", "RandomForest", "SVC"]
selected_models = st.sidebar.multiselect(
    "Selecciona uno o varios modelos",
    options=default_models,
    default=default_models
)

# =========================
# Generación de dataset
# =========================
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=n_redundant,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=[weight_majority, 1 - weight_majority],
    class_sep=class_sep,
    flip_y=flip_y,
    random_state=random_state
)

num_cols = [f"feat_{i}" for i in range(n_features)]
df = pd.DataFrame(X, columns=num_cols)
df["target"] = y

# Agregar una columna categórica sintética (derivada de feat_0)
bins = pd.qcut(df["feat_0"], q=3, labels=["low", "mid", "high"])
df["segment"] = bins.astype(str)  # categórica
# Reordenar columnas
df = df[num_cols + ["segment", "target"]]

st.subheader("📦 Dataset Sintético")
st.write(
    f"Registros: **{len(df)}** | Columnas (numéricas): **{n_features}** + **segment (categórica)** + **target (etiqueta)**"
)
st.dataframe(df.head(20))

# Botón para descargar CSV
st.download_button(
    "⬇️ Descargar dataset (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="synthetic_dataset.csv",
    mime="text/csv"
)

# =========================
# Visualizaciones exploratorias
# =========================
st.subheader("🔎 Exploración de datos")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Distribución de clases**")
    class_counts = df["target"].value_counts().sort_index().rename({0: "Clase 0", 1: "Clase 1"})
    fig_cls = px.bar(
        class_counts,
        title="Balance de clases",
        labels={"index": "Clase", "value": "Número de registros"}
    )
    st.plotly_chart(fig_cls, use_container_width=True)

with col2:
    st.markdown("**Mapa de calor de correlación (numéricas)**")
    corr = df[num_cols].corr()
    fig_corr, ax = plt.subplots()
    im = ax.imshow(corr, interpolation="nearest")
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=90)
    ax.set_yticks(range(len(num_cols)))
    ax.set_yticklabels(num_cols)
    ax.set_title("Correlación")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig_corr)

# Proyección PCA 2D para visualizar separabilidad
st.markdown("**Proyección PCA (2D)**")
# Preprocesamiento simple numérico para PCA
scaled_for_pca = StandardScaler().fit_transform(df[num_cols])
pca = PCA(n_components=2, random_state=random_state)
proj = pca.fit_transform(scaled_for_pca)
pca_df = pd.DataFrame(proj, columns=["PC1", "PC2"])
pca_df["target"] = df["target"].map({0: "Clase 0", 1: "Clase 1"})
fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="target", title="PCA 2D (solo numéricas)")
st.plotly_chart(fig_pca, use_container_width=True)

# =========================
# Preparación para modelado
# =========================
X = df.drop(columns=["target"])
y = df["target"]

numeric_features = num_cols
categorical_features = ["segment"]

transformers = []
if standardize:
    transformers.append(("num", StandardScaler(), numeric_features))
else:
    transformers.append(("num", "passthrough", numeric_features))

transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))

preprocess = ColumnTransformer(transformers=transformers)

# =========================
# Definir modelos
# =========================
model_map = {
    "GaussianNB": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=random_state),
    "SVC": SVC(probability=True, random_state=random_state)
}

# Validar selección
if not selected_models:
    st.warning("Selecciona al menos un modelo en la barra lateral.")
    st.stop()

# =========================
# Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# =========================
# Entrenamiento y evaluación
# =========================
st.header("🏁 Entrenamiento y evaluación")

results = []
tabs = st.tabs([f"🔬 {name}" for name in selected_models])

for tab, name in zip(tabs, selected_models):
    with tab:
        clf = model_map[name]
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        # Asegurar probas para ROC
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            # usar decision_function si existe, escalando a 0-1 aproximadamente
            if hasattr(pipe.named_steps["model"], "decision_function"):
                dec = pipe.decision_function(X_test)
                # min-max scaling
                y_proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
            else:
                # fallback uniforme
                y_proba = np.ones_like(y_pred, dtype=float) * (y_pred.mean())

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = np.nan

        results.append({"Modelo": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC AUC": auc})

        # Métricas
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1", f"{f1:.3f}")
        m5.metric("ROC AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")

        # Reporte de clasificación
        with st.expander("📄 Classification report"):
            st.text(classification_report(y_test, y_pred, digits=3))

        # Matriz de confusión
        st.markdown("**Matriz de confusión**")
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        fig_cm, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_xticks([0,1]); ax.set_xticklabels(["0","1"])
        ax.set_yticks([0,1]); ax.set_yticklabels(["0","1"])
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_title(f"Matriz de confusión — {name}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig_cm)

        # Curva ROC
        st.markdown("**Curva ROC**")
        fig_roc, ax = plt.subplots()
        try:
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
        except Exception:
            ax.text(0.5, 0.5, "No disponible", ha="center", va="center")
        ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
        ax.set_title(f"ROC — {name}")
        st.pyplot(fig_roc)

# =========================
# Ranking de modelos
# =========================
st.subheader("🏆 Comparativa de modelos")
res_df = pd.DataFrame(results).sort_values(by=["F1", "Accuracy"], ascending=False)
st.dataframe(res_df, use_container_width=True)

# =========================
# (Opcional) No supervisado: PCA y KMeans rápidos
# =========================
with st.expander("🌀 Extra (No supervisado): PCA + KMeans para exploración"):
    from sklearn.cluster import KMeans

    k = st.slider("Número de clusters (k)", min_value=2, max_value=8, value=2, step=1)
    scaler = StandardScaler()
    Z = scaler.fit_transform(df[num_cols])
    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(Z)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(pca2)
    unsup_df = pd.DataFrame(pca2, columns=["PC1", "PC2"])
    unsup_df["cluster"] = labels.astype(str)
    fig_unsup = px.scatter(unsup_df, x="PC1", y="PC2", color="cluster", title="Clusters (KMeans sobre PCA 2D)")
    st.plotly_chart(fig_unsup, use_container_width=True)

st.markdown("---")
st.caption("Hecho con ❤️ en Streamlit. Modifica parámetros en la barra lateral para experimentar.")

