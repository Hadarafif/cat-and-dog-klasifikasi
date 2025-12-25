from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import streamlit as st
from PIL import Image


# =========================
# Model & label utilities
# =========================
@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: Path):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path.as_posix())


def infer_model_spec(model) -> tuple[int, int]:
    """Return (img_size, channels) from model.input_shape; fallback (224,3)."""
    shp = model.input_shape
    if isinstance(shp, list):
        shp = shp[0]
    img_size, channels = 224, 3
    try:
        # (None, H, W, C)
        if len(shp) == 4 and isinstance(shp[1], int) and isinstance(shp[2], int) and shp[1] == shp[2]:
            img_size = int(shp[1])
            if shp[3] in (1, 3):
                channels = int(shp[3])
    except Exception:
        pass
    return img_size, channels


def load_label_map(models_dir: Path):
    path = models_dir / "labels.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def label_from_map(label_map, idx: int) -> str:
    if label_map is None:
        return str(idx)
    if isinstance(label_map, list) and 0 <= idx < len(label_map):
        return str(label_map[idx])
    if isinstance(label_map, dict):
        return str(label_map.get(str(idx), label_map.get(idx, idx)))
    return str(idx)


def preprocess(img: Image.Image, img_size: int, channels: int) -> np.ndarray:
    if channels == 1:
        img = img.convert("L").resize((img_size, img_size))
        arr = (np.array(img, dtype=np.float32) / 255.0)[..., None]  # (H,W,1)
    else:
        img = img.convert("RGB").resize((img_size, img_size))
        arr = np.array(img, dtype=np.float32) / 255.0  # (H,W,3)
    return np.expand_dims(arr, 0)  # (1,H,W,C)


def probs_from_pred(y: np.ndarray) -> np.ndarray:
    """Handle sigmoid/logits/softmax outputs -> probs (n_class,)."""
    y = np.squeeze(y)

    # binary sigmoid variants
    if y.ndim == 0:
        p = float(y)
        return np.array([1 - p, p], dtype=np.float32)
    if y.ndim == 1 and y.shape[0] == 1:
        p = float(y[0])
        return np.array([1 - p, p], dtype=np.float32)

    # multiclass (2 or more)
    probs = y.astype(np.float32)
    s = float(np.sum(probs))
    if not (0.98 <= s <= 1.02) or np.any(probs < 0):
        # treat as logits
        probs = np.exp(probs - np.max(probs))
        probs = probs / (np.sum(probs) + 1e-12)
    else:
        probs = probs / (s + 1e-12)
    return probs


# =========================
# Dataset helpers
# =========================
def list_images(folder: Path):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        files += list(folder.glob(ext))
    return sorted(files)


def dataset_available(dataset_dir: Path) -> bool:
    return (dataset_dir / "Cat").exists() and (dataset_dir / "Dog").exists()


# =========================
# UI: Prediction
# =========================
def ui_predict(models_dir: Path, dataset_dir: Path, model_specs: dict, label_map):
    st.subheader("🔮 Prediksi (pilih model + input gambar)")

    left, right = st.columns([1, 1], gap="large")

    with left:
        model_choice = st.selectbox(
            "Pilih Model",
            list(model_specs.keys()),
        )
        model_path = models_dir / model_specs[model_choice]
        if not model_path.exists():
            st.error(f"File model tidak ditemukan: {model_path.name}")
            st.stop()

        with st.spinner("Memuat model..."):
            model = load_keras_model(model_path)

        img_size_default, channels = infer_model_spec(model)

        st.caption("Model input spec terbaca otomatis dari input shape.")
        c1, c2, c3 = st.columns(3)
        with c1:
            img_size = st.number_input("IMG_SIZE", value=int(img_size_default), step=1)
        with c2:
            st.write("channels")
            st.selectbox("channels (auto)", [channels], index=0, disabled=True)
        with c3:
            topk = st.slider("Top-K", 1, 5, 3)

        st.markdown("---")
        mode = st.radio(
            "Sumber gambar",
            ["Upload gambar"] + (["Pilih dari dataset contoh (Cat/Dog)"] if dataset_available(dataset_dir) else []),
            horizontal=True,
        )

        img = None
        caption = ""

        if mode.startswith("Pilih") and dataset_available(dataset_dir):
            cls = st.selectbox("Kelas (folder)", ["Cat", "Dog"])
            files = list_images(dataset_dir / cls)
            if not files:
                st.warning(f"Tidak ada gambar di: {dataset_dir/cls}")
                st.stop()
            pick = st.radio("Pilih file", ["Acak", "Manual"], horizontal=True)
            chosen = random.choice(files) if pick == "Acak" else st.selectbox("File", files, format_func=lambda p: p.name)
            img = Image.open(chosen)
            caption = f"{cls}/{chosen.name}"
        else:
            up = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg", "webp"])
            if up is None:
                st.info("Upload gambar dulu untuk mulai prediksi.")
                st.stop()
            img = Image.open(up)
            caption = "Uploaded"

    with right:
        st.subheader("🖼️ Preview")
        st.image(img, caption=caption, use_container_width=True)

        st.write("")
        if st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True):
            x = preprocess(img, int(img_size), int(channels))
            y = model.predict(x, verbose=0)
            probs = probs_from_pred(y)

            order = np.argsort(-probs)[:topk]
            best = int(order[0])

            st.subheader("✅ Hasil Prediksi")
            st.success(f"Model: **{model_choice}**")
            st.success(f"Prediksi: **{label_from_map(label_map, best)}** (prob: **{float(probs[best]):.4f}**)")

            import pandas as pd
            rows = [
                {"class_id": int(i), "label": label_from_map(label_map, int(i)), "prob": float(probs[i])}
                for i in order
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# =========================
# UI: Evaluation (3 models)
# =========================
def ui_evaluate(models_dir: Path, dataset_dir: Path, model_specs: dict, label_map):
    st.subheader("📊 Evaluasi & Analisis Performa (3 Model)")

    st.info(
        "Sesuai ketentuan: praktikan wajib evaluasi & analisis performa untuk:\n"
        "- Neural Network base (non-pretrained)\n"
        "- Model Pretrained 1\n"
        "- Model Pretrained 2\n"
    )

    if not dataset_available(dataset_dir):
        st.error("Dataset contoh tidak ditemukan. Pastikan ada folder: `models/citra dataset contoh/Cat` dan `Dog`.")
        st.stop()

    # gather dataset
    cat_files = list_images(dataset_dir / "Cat")
    dog_files = list_images(dataset_dir / "Dog")

    if len(cat_files) == 0 or len(dog_files) == 0:
        st.error("Folder Cat/Dog kosong. Isi dulu gambar untuk evaluasi.")
        st.stop()

    max_per_class = st.slider("Maks gambar per kelas (untuk evaluasi cepat)", 5, 200, 40, step=5)
    cat_files = cat_files[:max_per_class]
    dog_files = dog_files[:max_per_class]

    # label index from labels.json expected: 0=Cat 1=Dog
    # build name_to_id from label_map
    name_to_id = {}
    if isinstance(label_map, dict):
        # {"0":"Cat","1":"Dog"} -> {"Cat":0,"Dog":1}
        for k, v in label_map.items():
            try:
                name_to_id[str(v)] = int(k)
            except Exception:
                pass

    if "Cat" not in name_to_id or "Dog" not in name_to_id:
        st.warning("labels.json tidak terbaca sempurna. Pastikan format: {'0':'Cat','1':'Dog'}")
        # fallback assume Cat=0 Dog=1
        name_to_id = {"Cat": 0, "Dog": 1}

    do_run = st.button("🧪 Jalankan Evaluasi 3 Model", type="primary")
    if not do_run:
        st.caption("Klik tombol di atas untuk menghitung accuracy, confusion matrix, dan classification report.")
        return

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import pandas as pd

    results = []

    prog = st.progress(0, text="Mulai evaluasi...")
    total_models = len(model_specs)

    for mi, (title, fname) in enumerate(model_specs.items(), start=1):
        model_path = models_dir / fname
        if not model_path.exists():
            results.append({"Model": title, "File": fname, "Accuracy": None, "Note": "File not found"})
            continue

        with st.spinner(f"Evaluasi: {title}"):
            m = load_keras_model(model_path)
            img_size_m, channels_m = infer_model_spec(m)

            y_true = []
            y_pred = []

            # Cat
            for fp in cat_files:
                im = Image.open(fp)
                x = preprocess(im, img_size_m, channels_m)
                prob = probs_from_pred(m.predict(x, verbose=0))
                pred = int(np.argmax(prob))
                y_true.append(name_to_id["Cat"])
                y_pred.append(pred)

            # Dog
            for fp in dog_files:
                im = Image.open(fp)
                x = preprocess(im, img_size_m, channels_m)
                prob = probs_from_pred(m.predict(x, verbose=0))
                pred = int(np.argmax(prob))
                y_true.append(name_to_id["Dog"])
                y_pred.append(pred)

            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=[name_to_id["Cat"], name_to_id["Dog"]])
            rep = classification_report(
                y_true,
                y_pred,
                target_names=["Cat", "Dog"],
                digits=4,
                zero_division=0,
            )

            results.append(
                {
                    "Model": title,
                    "File": fname,
                    "Accuracy": float(acc),
                    "Confusion Matrix": cm,
                    "Report": rep,
                    "Note": "",
                }
            )

        prog.progress(int(mi / total_models * 100), text=f"Selesai {mi}/{total_models}")

    st.progress(100, text="Evaluasi selesai ✅")

    # Summary table
    st.markdown("### 📌 Ringkasan Performa")
    df = pd.DataFrame(
        [{"Model": r["Model"], "File": r["File"], "Accuracy": r["Accuracy"], "Note": r.get("Note", "")} for r in results]
    )
    st.dataframe(df.sort_values("Accuracy", ascending=False, na_position="last"), use_container_width=True)

    # Detailed
    st.markdown("---")
    st.markdown("### 🔎 Detail Evaluasi per Model")
    for r in results:
        st.markdown(f"#### {r['Model']}")
        st.caption(r["File"])
        if r["Accuracy"] is None:
            st.error(f"Gagal evaluasi: {r.get('Note','')}")
            continue

        st.metric("Accuracy", f"{r['Accuracy']:.4f}")
        st.write("Confusion Matrix (rows=true, cols=pred) urutan: [Cat, Dog]")
        st.write(r["Confusion Matrix"])
        st.code(r["Report"], language="text")
        st.markdown("---")


# =========================
# Entrypoint for app.py
# =========================
def run_citra_app(models_dir: Path, page: str):
    """
    models_dir/
      labels.json
      model_base_non_pretrained.keras
      model_pretrained_1_mobilenetv2.keras
      model_pretrained_2_efficientnetb0.keras
      citra dataset contoh/Cat, Dog
    """
    dataset_dir = models_dir / "citra dataset contoh"
    label_map = load_label_map(models_dir)

    model_specs = {
        "Neural Network base (non-pretrained)": "model_base_non_pretrained.keras",
        "Model Pretrained 1": "model_pretrained_1_mobilenetv2.keras",
        "Model Pretrained 2": "model_pretrained_2_efficientnetb0.keras",
    }

    if label_map is None:
        st.warning("labels.json tidak ditemukan. Label akan tampil sebagai index (0/1). Disarankan buat labels.json.")

    if page == "Prediksi":
        ui_predict(models_dir=models_dir, dataset_dir=dataset_dir, model_specs=model_specs, label_map=label_map)
    else:
        ui_evaluate(models_dir=models_dir, dataset_dir=dataset_dir, model_specs=model_specs, label_map=label_map)
