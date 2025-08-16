import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(
    page_title="Comparador de Dosis en Procedimientos Angiográficos",
    page_icon="🩻",
    layout="wide"
)

st.title("🩻 Comparador de Dosis entre Procedimientos Angiográficos")
st.caption("App educativa para análisis y comparación de dosis. No sustituye mediciones clínicas ni normativa local.")

st.markdown("---")

# -----------------------------
# Dataset de ejemplo (si no se sube CSV)
# -----------------------------
def ejemplo_df():
    np.random.seed(42)
    procedimientos = [
        "Angiografía cerebral diagnóstica",
        "Coiling de aneurisma",
        "MAV (embolización)",
        "Trombectomía mecánica",
        "Angioplastia periférica"
    ]
    n = 220
    df = pd.DataFrame({
        "procedimiento": np.random.choice(procedimientos, n, p=[0.25, 0.2, 0.2, 0.2, 0.15]),
        "DAP_Gycm2": np.round(np.random.gamma(shape=3.5, scale=6.0, size=n), 2),   # Gy·cm²
        "Ka_r_mGy": np.round(np.random.gamma(shape=5.0, scale=35.0, size=n), 1),   # Air kerma ref, mGy
        "tiempo_fluoro_min": np.round(np.random.gamma(shape=2.5, scale=6.0, size=n), 1),
    })
    # Opcional: simular atípicos
    outlier_idx = np.random.choice(df.index, 5, replace=False)
    df.loc[outlier_idx, "DAP_Gycm2"] *= 3
    df.loc[outlier_idx, "Ka_r_mGy"] *= 2.2
    df.loc[outlier_idx, "tiempo_fluoro_min"] *= 1.8
    return df

# -----------------------------
# Carga de datos
# -----------------------------
st.sidebar.header("📥 Datos")
archivo = st.sidebar.file_uploader(
    "Sube tu CSV (columnas esperadas: procedimiento, DAP_Gycm2, Ka_r_mGy, tiempo_fluoro_min)",
    type=["csv"]
)

if archivo is not None:
    df = pd.read_csv(archivo)
    fuente = "CSV subido"
else:
    df = ejemplo_df()
    fuente = "Dataset de ejemplo"

st.sidebar.info(f"Fuente de datos: **{fuente}** | Registros: **{len(df)}**")

# Validación mínima de columnas
cols_req = {"procedimiento", "DAP_Gycm2", "Ka_r_mGy", "tiempo_fluoro_min"}
faltantes = cols_req - set(df.columns)
if faltantes:
    st.error(f"Faltan columnas obligatorias en el dataset: {faltantes}")
    st.stop()

# -----------------------------
# Filtros
# -----------------------------
st.subheader("🔎 Filtros")
col_f1, col_f2 = st.columns([2, 1])

with col_f1:
    procs = sorted(df["procedimiento"].dropna().unique().tolist())
    sel_procs = st.multiselect("Procedimientos a incluir", procs, default=procs)

with col_f2:
    # Rango por tiempo (simple)
    tmin, tmax = float(df["tiempo_fluoro_min"].min()), float(df["tiempo_fluoro_min"].max())
    rango_t = st.slider("Rango de tiempo de fluoroscopía (min)", min_value=0.0, max_value=max(1.0, round(tmax,1)),
                        value=(0.0, round(tmax,1)), step=0.5)

df_f = df[df["procedimiento"].isin(sel_procs)].copy()
df_f = df_f[(df_f["tiempo_fluoro_min"] >= rango_t[0]) & (df_f["tiempo_fluoro_min"] <= rango_t[1])]

st.caption(f"Registros filtrados: **{len(df_f)}**")

# -----------------------------
# Resumen por procedimiento
# -----------------------------
def resumen(df_in, var):
    g = df_in.groupby("procedimiento")[var]
    out = pd.DataFrame({
        "n": g.count(),
        "media": g.mean().round(2),
        "mediana": g.median().round(2),
        "p75 (DRL candidato)": g.quantile(0.75).round(2),
        "std": g.std(ddof=1).round(2),
        "min": g.min().round(2),
        "max": g.max().round(2),
    }).reset_index()
    return out

st.subheader("📊 Resumen estadístico por procedimiento")

tabs = st.tabs(["DAP (Gy·cm²)", "Ka,r (mGy)", "Tiempo de fluoroscopía (min)"])

with tabs[0]:
    res_dap = resumen(df_f, "DAP_Gycm2")
    st.dataframe(res_dap, use_container_width=True)
with tabs[1]:
    res_kar = resumen(df_f, "Ka_r_mGy")
    st.dataframe(res_kar, use_container_width=True)
with tabs[2]:
    res_t = resumen(df_f, "tiempo_fluoro_min")
    st.dataframe(res_t, use_container_width=True)

# -----------------------------
# Gráficos (Matplotlib)
# -----------------------------
def boxplot_por_proc(df_in, var, ylabel):
    procs_ord = df_in.groupby("procedimiento")[var].median().sort_values().index.tolist()
    data = [df_in[df_in["procedimiento"] == p][var].values for p in procs_ord]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, labels=procs_ord, showfliers=True)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Distribución por procedimiento - {var}")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    return fig

def barras_media_std(df_in, var, ylabel):
    agg = df_in.groupby("procedimiento")[var].agg(["mean", "std", "count"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(agg.index, agg["mean"])
    # barras de error (std)
    ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], fmt="none", capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Media ± DE por procedimiento - {var}")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    return fig

st.subheader("📈 Visualizaciones")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Boxplot**")
    st.pyplot(boxplot_por_proc(df_f, "DAP_Gycm2", "DAP (Gy·cm²)"), clear_figure=True)
    st.pyplot(boxplot_por_proc(df_f, "Ka_r_mGy", "Ka,r (mGy)"), clear_figure=True)
    st.pyplot(boxplot_por_proc(df_f, "tiempo_fluoro_min", "Tiempo (min)"), clear_figure=True)

with c2:
    st.markdown("**Barras (Media ± DE)**")
    st.pyplot(barras_media_std(df_f, "DAP_Gycm2", "DAP (Gy·cm²)"), clear_figure=True)
    st.pyplot(barras_media_std(df_f, "Ka_r_mGy", "Ka,r (mGy)"), clear_figure=True)
    st.pyplot(barras_media_std(df_f, "tiempo_fluoro_min", "Tiempo (min)"), clear_figure=True)

# -----------------------------
# Descargas
# -----------------------------
st.subheader("💾 Exportar datos")
col_d1, col_d2, col_d3 = st.columns(3)

def make_download_button(df_in, filename, label):
    buf = io.StringIO()
    df_in.to_csv(buf, index=False)
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="text/csv"
    )

with col_d1:
    make_download_button(df_f, "datos_filtrados.csv", "Descargar datos filtrados (CSV)")
with col_d2:
    make_download_button(res_dap, "resumen_DAP.csv", "Descargar resumen DAP")
with col_d3:
    make_download_button(res_kar, "resumen_Ka_r.csv", "Descargar resumen Ka,r")

st.markdown("---")
with st.expander("Notas y buenas prácticas"):
    st.markdown(
        """
- **Columnas esperadas en tu CSV**:  
  `procedimiento`, `DAP_Gycm2`, `Ka_r_mGy`, `tiempo_fluoro_min`  
- El **percentil 75 (p75)** suele emplearse como **candidato a DRL** a nivel poblacional (referencia educativa).
- Verifica unidades: DAP en **Gy·cm²**, Ka,r en **mGy**, tiempo en **min**.
- Si tu CSV incluye más variables (edad, sexo, operador, equipo), puedes ampliarlo en el código para filtrar/estratificar.
        """
    )
