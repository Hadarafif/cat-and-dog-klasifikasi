<h1 align="center">🐶🐱 Dog vs Cat Image Classification</h1>

<p align="center">
  <b>Ujian Akhir Praktikum (UAP) – Pembelajaran Mesin</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/STATUS-COMPLETED-brightgreen" />
  <img src="https://img.shields.io/badge/PYTHON-3.10-blue" />
  <img src="https://img.shields.io/badge/TENSORFLOW-2.x-orange" />
  <img src="https://img.shields.io/badge/STREAMLIT-FRAMEWORK-red" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c599b23b-4ad1-444e-9377-20d1b6017733" width="275" height="183">
</p>

Project ini dibuat untuk memenuhi **Ujian Akhir Praktikum (UAP)** pada mata kuliah **Pembelajaran Mesin / Machine Learning**.  
Fokus project adalah membangun sistem **klasifikasi citra** untuk membedakan **gambar anjing (Dog)** dan **kucing (Cat)** menggunakan **Deep Learning** serta membandingkan performa beberapa model.

---

## 📌 Deskripsi Project

Klasifikasi citra merupakan salah satu penerapan utama dalam bidang **Computer Vision**.  
Pada project ini, dikembangkan sebuah aplikasi berbasis **Streamlit** yang mampu:
- Menerima input gambar dari pengguna
- Melakukan prediksi kelas **Dog / Cat**
- Menampilkan hasil prediksi dan **evaluasi performa model**

Project ini mengimplementasikan **3 model wajib** sesuai ketentuan UAP:
1. Base CNN (Non-pretrained)
2. Pretrained Model 1 (MobileNetV2)
3. Pretrained Model 2 (EfficientNetB0)

---

## 🎯 Tujuan Project

Tujuan dari project ini adalah:
1. Mengimplementasikan **klasifikasi citra Dog vs Cat**
2. Membandingkan performa model **non-pretrained vs pretrained**
3. Menyediakan aplikasi interaktif sebagai media evaluasi model
4. Memenuhi seluruh ketentuan **Ujian Akhir Praktikum (UAP)**

---

## 📂 Struktur Repository

```text
dog-vs-cat-klasifikasi/
├── models/              # Folder model (tidak disertakan di GitHub)
├── src/                 # Source code pendukung
├── app.py               # Aplikasi Streamlit
├── UAP.ipynb            # Notebook training & evaluasi
├── pyproject.toml       # Konfigurasi environment (PDM)
├── pdm.lock             # Lock dependency
├── .gitignore           # Ignore file besar (model)
└── README.md            # Dokumentasi project
```
---

## 📊 Dataset

Dataset yang digunakan adalah **dataset citra Dog vs Cat** yang terdiri dari dua kelas:
- **Cat**
- **Dog**

Tahapan preprocessing meliputi:
- Resize gambar ke ukuran 224×224
- Normalisasi pixel
- Pembagian data training dan validation

---

## 🧠 Metode & Model

### 🔹 Model yang Digunakan
- **Base CNN (Non-pretrained)**
- **MobileNetV2 (Pretrained)**
- **EfficientNetB0 (Pretrained)**

Framework:
- TensorFlow / Keras

---

## 🖥️ Dashboard Aplikasi (Streamlit)

Aplikasi Streamlit menyediakan fitur:
- Pemilihan model
- Upload gambar atau dataset contoh
- Prediksi kelas Dog / Cat
- Evaluasi performa (akurasi, confusion matrix, classification report)

### 📸 Tampilan Dashboard
<img width="2015" height="964" alt="{1EEB4143-7AEE-45FF-9CB1-9435ABE1F893}" src="https://github.com/user-attachments/assets/d5ae27de-5e04-4e97-9479-4b8714d81d3f" />


## 📈 Hasil Training (Base CNN – Non-pretrained)

### Grafik Akurasi & Loss
![Accuracy & Loss]
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/9698e51b-627a-4b7b-91f8-d596e3d94d32" />


Model Base CNN menunjukkan peningkatan akurasi secara bertahap dengan loss yang semakin menurun seiring epoch.

---

## 📊 Evaluasi Model (Classification Report)

### 🔹 Base CNN (Non-pretrained)
Accuracy: 80%
| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| Cat  | 0.78      | 0.82   |  0.80    |
| Dog  | 0.81      | 0.77   |  0.79    |

Accuracy: 80%
### 🔹 Pretrained 1 – MobileNetV2
| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| Cat  |      0.99 | 0.96   |   0.98   |
| Dog  |      0.97 | 0.99   |   0.98   |
Accuracy: 98%
📌 **Model terbaik berdasarkan evaluasi**  
MobileNetV2 memberikan performa paling optimal dengan akurasi dan F1-score tinggi pada kedua kelas.


### 🔹 Pretrained 2 – EfficientNetB0

Model ini mengalami **kegagalan klasifikasi** (bias ke satu kelas), ditunjukkan oleh:
- Recall Cat = 0.00
- Recall Dog = 1.00

Hal ini menunjukkan bahwa model belum terkonvergensi dengan baik atau terdapat isu pada proses training / preprocessing.

---

## 🚀 Instalasi Environment

Project menggunakan **PDM**.

```bash
pdm install
▶️ Menjalankan Aplikasi
pdm run streamlit run app.py
```
📥 Download Model

File model berukuran besar (>100MB) sehingga tidak disimpan di GitHub.

🔗 Link Google Drive Model:

([https://drive.google.com/drive/folders/1fECM1jR9D_XO2yIt9cHHE5MzwntO6Qc3?usp=sharing])

Cara menggunakan:

Download file model

Simpan ke folder:
models/

🔗 Link Dataset:
([https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset])



Jalankan aplikasi Streamlit
---
👤 Identitas Mahasiswa

Nama: M. Haidar Afif Al Azizi
NIM: 202210370311191
Program Studi: Informatika
Universitas: Universitas Muhammadiyah Malang
