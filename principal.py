import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
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
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification

# Viz
import matplotlib.pyplot as plt
import plotly.express as px

# =========================
# Configuración de página
# =========================
st.set_page_config(page_title="ML Playground: CSV + Sintético", layout="wide")
st.title("🧠 Machine Learning Playground")
st.caption("Clasificación supervisada con **upload de CSV** o **datos sintéticos**. Requisitos del CSV: **≥ 300 filas** y **≥ 6 columnas**.")

# =========================
# Funciones auxiliares
# =========================
def validate_dataset(df: pd.DataFrame) -> tuple[bool, str]:
    if df is None or df.empty:
        return False, "El archivo está vacío."
    # Eliminar columnas totalmente vacías
    df = df.dropna(axis=1, how="all")
    if df.shape[0] < 300:
        return False, f"El dataset tiene {df.shape[0]} filas (< 300)."
    if df.shape[1] < 6:
        return False, f"El dataset tiene {df.shape[1]} columnas (< 6)."
    return True, ""

def split_types(df: pd.DataFrame, target_col: str):
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    return feature_cols, num_cols, cat_cols

def get_models(random_state: int):
    return {
        "GaussianNB": GaussianNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None, solver="lbfgs"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "SVC": SVC(probability=True, random_state=random_state)
    }

def build_preprocessor(num_cols, cat_cols, standardize=True):
    num_pipe = []
    if standardize:
        num_pipe = [("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())]
    else:
        num_pipe = [("imputer", SimpleImputer(strategy="median"))]

    cat_pipe = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # OneHotEncoder denso para compatibilidad con GaussianNB
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_pipe), num_cols),
            ("cat", Pipeline(cat_pipe), cat_cols),
        ],
        remainder="drop"  # sólo usar columnas seleccionadas
    )
    return pre

def compute_metrics(y_true, y_pred, y_proba=None):
    # Soporte multi-clase con promedios ponderados
    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    auc = np.nan
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = np.nan
    return acc, prec, rec, f1, auc

# =========================
# Sidebar — parámetros
# =========================
st.sidebar.header("📥 Datos de entrada")
mode = st.sidebar.radio("Fuente de datos", ["Subir CSV", "Sintético"], index=0)

st.sidebar.markdown("### ⚙️ Parámetros generales")
random_state = st.sidebar.number_input("Random state", min_value=0, value=42)
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
standardize = st.sidebar.checkbox("Estandarizar variables numéricas", value=True)

st.sidebar.markdown("### 🤖 Modelos a entrenar")
default_models = ["GaussianNB", "LogisticRegression", "RandomForest", "SVC"]
selected_models = st.sidebar.multiselect(
    "Selecciona uno o varios modelos",
    options=default_models,
    default=default_models
)

# =========================
# Carga o generación de datos
# =========================
df = None
target_col = None
num_cols = []
cat_cols = []

if mode == "Subir CSV":
    st.subheader("📥 Subir archivo CSV")
    with st.expander("📄 Estándares del archivo (obligatorio)", expanded=True):
        st.markdown(
            """
**Tu CSV debe cumplir:**
- Tener **≥ 300 filas** (registros).
- Tener **≥ 6 columnas** (features + posibles auxiliares).
- Incluir **al menos una columna objetivo (label)** para clasificación.
- **Encabezado** en la primera fila.
- **Codificación UTF-8** y **separador coma (,)**.
- Valores faltantes permitidos, pero se imputarán automáticamente.

> Sugerencia: si tu objetivo es binario, usa valores claros como `0/1`, `Yes/No` o `True/False`.
"""
        )

    uploaded = st.file_uploader("Arrastra o selecciona tu archivo CSV", type=["csv"])
    if uploaded is not None:
        try:
            data = uploaded.read().decode("utf-8")
            df = pd.read_csv(StringIO(data))
        except UnicodeDecodeError:
            # Reintento común con latin-1
            uploaded.seek(0)
            data = uploaded.read().decode("latin-1")
            df = pd.read_csv(StringIO(data))

        ok, msg = validate_dataset(df)
        if not ok:
            st.error(f"❌ {msg}")
            st.stop()

        st.success(f"✅ CSV cargado: {df.shape[0]} filas, {df.shape[1]} columnas.")
        st.dataframe(df.head(20), use_container_width=True)

        # Selección de columna objetivo
        st.markdown("#### 🎯 Selecciona la **columna objetivo** (label) para clasificación")
        target_col = st.selectbox("Columna objetivo", options=df.columns)

        if target_col:
            # Tipos
            feature_cols, num_cols, cat_cols = split_types(df, target_col)

            # Permite elegir qué columnas usar como features (opcional)
            with st.expander("🧩 Selección de columnas de entrada (opcional)", expanded=False):
                chosen_features = st.multiselect(
                    "Elige columnas para usar como **features** (si no seleccionas, se usan todas excepto el objetivo)",
                    options=feature_cols,
                    default=feature_cols
                )
                feature_cols = chosen_features if len(chosen_features) >= 1 else feature_cols
                num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
                cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

            # Validaciones de target
            nunique = df[target_col].nunique(dropna=True)
            if nunique < 2:
                st.error("❌ La columna objetivo debe tener al menos **2 clases**.")
                st.stop()

            st.info(
                f"Features seleccionadas: **{len(feature_cols)}** "
                f"(numéricas: {len(num_cols)}, categóricas: {len(cat_cols)}). "
                f"Clases en objetivo: **{nunique}**."
            )

            # Visualizaciones exploratorias (si hay suficientes numéricas)
            st.subheader("🔎 Exploración de datos (CSV)")
            if len(num_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Distribución de la variable objetivo**")
                    class_counts = df[target_col].value_counts(dropna=False)
                    fig_cls = px.bar(
                        class_counts,
                        title="Conteo por clase",
                        labels={"index": "Clase", "value": "Número de registros"}
                    )
                    st.plotly_chart(fig_cls, use_container_width=True)

                with col2:
                    st.markdown("**Mapa de calor de correlación (numéricas)**")
                    corr = df[num_cols].corr(numeric_only=True)
                    fig_corr, ax = plt.subplots()
                    im = ax.imshow(corr, interpolation="nearest")
                    ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=90)
                    ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
                    ax.set_title("Correlación")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig_corr)

                # PCA 2D (solo numéricas)
                st.markdown("**Proyección PCA (2D) con columnas numéricas**")
                try:
                    scaled_for_pca = StandardScaler().fit_transform(df[num_cols])
                    pca = PCA(n_components=2, random_state=random_state)
                    proj = pca.fit_transform(scaled_for_pca)
                    pca_df = pd.DataFrame(proj, columns=["PC1", "PC2"])
                    pca_df["target"] = df[target_col].astype(str)
                    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="target", title="PCA 2D (solo numéricas)")
                    st.plotly_chart(fig_pca, use_container_width=True)
                except Exception as e:
                    st.warning(f"No se pudo calcular PCA: {e}")

else:
    # =========================
    # Sintético (por compatibilidad)
    # =========================
    st.subheader("🎲 Datos sintéticos")
    n_samples = st.slider("Número de registros (≥ 300)", min_value=300, max_value=10000, value=600, step=100)
    n_features = st.slider("Número de columnas numéricas", min_value=6, max_value=20, value=6, step=1)
    n_informative = st.slider("Características informativas", min_value=2, max_value=n_features, value=min(4, n_features), step=1)
    n_redundant = st.slider("Características redundantes", min_value=0, max_value=max(0, n_features-1), value=1, step=1)
    class_sep = st.slider("Separación entre clases", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    flip_y = st.slider("Ruido (flip_y)", min_value=0.0, max_value=0.3, value=0.02, step=0.01)
    weight_majority = st.slider("Proporción clase 0 (balance de clases)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

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
    bins = pd.qcut(df["feat_0"], q=3, labels=["low", "mid", "high"])
    df["segment"] = bins.astype(str)
    df = df[num_cols + ["segment", "target"]]

    st.write(f"Registros: **{len(df)}** | Columnas (numéricas): **{n_features}** + **segment** + **target**")
    st.dataframe(df.head(20), use_container_width=True)

    # Descarga
    st.download_button(
        "⬇️ Descargar dataset sintético (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="synthetic_dataset.csv",
        mime="text/csv"
    )

    # Preparar nombres para modelado
    target_col = "target"
    # Recalcular tipos con target fijo
    _, num_cols, cat_cols = split_types(df, target_col)

# =========================
# Preparación para modelado (si hay df y target)
# =========================
if df is not None and target_col is not None:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Armamos preprocesamiento
    preprocess = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols, standardize=standardize)

    # =========================
    # Train/Test split
    # =========================
    try:
        stratify_param = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
    except ValueError as e:
        st.error(f"No se pudo realizar el split de train/test: {e}")
        st.stop()

    # =========================
    # Entrenamiento y evaluación
    # =========================
    if not selected_models:
        st.warning("Selecciona al menos un modelo en la barra lateral para continuar.")
        st.stop()

    st.header("🏁 Entrenamiento y evaluación")
    model_map = get_models(random_state)
    results = []
    tabs = st.tabs([f"🔬 {name}" for name in selected_models])

    is_binary = (y_train.nunique() == 2)

    for tab, name in zip(tabs, selected_models):
        with tab:
            clf = model_map[name]
            pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)

            # Obtener probabilidades o scores para ROC (solo binario)
            y_proba = None
            if is_binary:
                if hasattr(pipe.named_steps["model"], "predict_proba"):
                    y_proba = pipe.predict_proba(X_test)[:, 1]
                elif hasattr(pipe.named_steps["model"], "decision_function"):
                    dec = pipe.decision_function(X_test)
                    y_proba = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)

            acc, prec, rec, f1, auc = compute_metrics(y_test, y_pred, y_proba)
            results.append({"Modelo": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC AUC": auc})

            # Métricas rápidas
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("Precision", f"{prec:.3f}")
            m3.metric("Recall", f"{rec:.3f}")
            m4.metric("F1", f"{f1:.3f}")
            m5.metric("ROC AUC", f"{auc:.3f}" if is_binary and not np.isnan(auc) else "N/A")

            # Reporte de clasificación
            with st.expander("📄 Classification report"):
                # Para multi-clase se muestra completo
                st.text(classification_report(y_test, y_pred, digits=3))

            # Matriz de confusión
            st.markdown("**Matriz de confusión**")
            labels_sorted = sorted(y.unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
            fig_cm, ax = plt.subplots()
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
            ax.set_xticks(range(len(labels_sorted))); ax.set_xticklabels(labels_sorted, rotation=0)
            ax.set_yticks(range(len(labels_sorted))); ax.set_yticklabels(labels_sorted)
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center")
            ax.set_title(f"Matriz de confusión — {name}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig_cm)

            # Curva ROC (solo binario)
            st.markdown("**Curva ROC**" + ("" if is_binary else " (solo disponible en binario)"))
            fig_roc, ax = plt.subplots()
            if is_binary and y_proba is not None:
                try:
                    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
                    ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
                    ax.set_title(f"ROC — {name}")
                except Exception:
                    ax.text(0.5, 0.5, "No disponible", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "No disponible para multi-clase", ha="center", va="center")
            st.pyplot(fig_roc)

    # =========================
    # Ranking de modelos
    # =========================
    st.subheader("🏆 Comparativa de modelos")
    res_df = pd.DataFrame(results).sort_values(by=["F1", "Accuracy"], ascending=False)
    st.dataframe(res_df, use_container_width=True)

    # =========================
    # Extra no supervisado (PCA+KMeans) si hay >=2 numéricas
    # =========================
    with st.expander("🌀 Extra (No supervisado): PCA + KMeans para exploración"):
        if len(num_cols) < 2:
            st.warning("Se requieren al menos 2 columnas numéricas para PCA/KMeans.")
        else:
            k = st.slider("Número de clusters (k)", min_value=2, max_value=8, value=2, step=1)
            try:
                scaler = StandardScaler()
                Z = scaler.fit_transform(df[num_cols])
                pca2 = PCA(n_components=2, random_state=random_state).fit_transform(Z)
                km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
                labels = km.fit_predict(pca2)
                unsup_df = pd.DataFrame(pca2, columns=["PC1", "PC2"])
                unsup_df["cluster"] = labels.astype(str)
                fig_unsup = px.scatter(unsup_df, x="PC1", y="PC2", color="cluster",
                                       title="Clusters (KMeans sobre PCA 2D)")
                st.plotly_chart(fig_unsup, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo ejecutar PCA/KMeans: {e}")

st.markdown("---")
st.caption("Hecho con ❤️ en Streamlit. Cambia la fuente de datos en la barra lateral para probar con CSV o datos sintéticos.")
