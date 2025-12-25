from pathlib import Path
import streamlit as st

from src.infer_citra import run_citra_app

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

st.set_page_config(
    page_title="Cat vs Dog • Image Classifier",
    page_icon="🐾",
    layout="wide",
)

# ---------- Style (simple, clean, nice) ----------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
      .hero {
        padding: 1.1rem 1.25rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,.12);
        background: linear-gradient(135deg, rgba(59,130,246,.15), rgba(16,185,129,.10));
        box-shadow: 0 10px 30px rgba(0,0,0,.18);
      }
      .hero h1 { margin: 0; font-size: 1.8rem; }
      .hero p { margin: .35rem 0 0; opacity: .9; }
      .chip {
        display:inline-block; padding:0.25rem 0.6rem; border-radius:999px;
        background: rgba(56,189,248,.12); border: 1px solid rgba(56,189,248,.25);
        font-size: 0.8rem;
      }
      .card {
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(255,255,255,.03);
        height: 100%;
      }
      .muted { opacity: .85; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.06); }
      div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.markdown("## 🐾 Cat vs Dog")
st.sidebar.caption("Sistem Website Sederhana (Streamlit) untuk demo 3 model citra.")
st.sidebar.markdown("---")

st.sidebar.markdown("**📌 Ketentuan tugas**")
st.sidebar.markdown(
    """
- Input data dari pengguna ✅  
- Tampilkan hasil prediksi sesuai pemilihan model ✅  
- Evaluasi & analisis performa 3 model ✅  
    """
)

st.sidebar.markdown("---")
page = st.sidebar.radio("Menu", ["Prediksi", "Evaluasi & Analisis"], index=0)

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
      <span class="chip">Image Classification</span>
      <h1>🐶🐱 Cat vs Dog Classifier</h1>
      <p class="muted">Pilih model → input gambar → lihat prediksi & evaluasi performa.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# cards
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="card">
          <div class="chip">Model</div>
          <h3 style="margin:.5rem 0 0;">3 Model Wajib</h3>
          <p class="muted" style="margin:.35rem 0 0;">
            Base (non-pretrained), Pretrained 1, Pretrained 2.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="card">
          <div class="chip">Input</div>
          <h3 style="margin:.5rem 0 0;">Upload / Dataset</h3>
          <p class="muted" style="margin:.35rem 0 0;">
            Upload gambar atau pilih dari folder dataset contoh.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="card">
          <div class="chip">Evaluasi</div>
          <h3 style="margin:.5rem 0 0;">Akurasi + CM + Report</h3>
          <p class="muted" style="margin:.35rem 0 0;">
            Bandingkan performa ketiga model pada dataset contoh.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

run_citra_app(
    models_dir=MODELS_DIR,
    page=page,
)
