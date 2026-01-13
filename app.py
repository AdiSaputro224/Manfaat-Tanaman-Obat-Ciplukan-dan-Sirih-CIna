# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Manfaat Tanaman Obat Ciplukan & Sirih Cina",
    layout="wide"
)

DATA_PATH = "dataset_tanaman_obat_ilmiah_1000.csv"

# -----------------------------------
# LOAD DATASET
# -----------------------------------
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)

    expected_cols = [
        "tanaman",
        "gejala",
        "manfaat",
        "efek_samping",
        "referensi"
    ]

    for col in expected_cols:
        if col not in df.columns:
            st.error(f"Kolom '{col}' tidak ditemukan dalam dataset.")
            st.stop()

    # Normalisasi teks
    for c in expected_cols:
        df[c] = df[c].astype(str).fillna("").str.strip()

    return df

# -----------------------------------
# PARSING GEJALA
# -----------------------------------
def split_gejala(text):
    if text.strip() == "":
        return []
    text = text.lower()
    text = re.sub(r"\s+dan\s+", ",", text)
    parts = re.split(r"[;,/|]+", text)
    return [p.strip() for p in parts if p.strip()]

# -----------------------------------
# BUILD DATASET MACHINE LEARNING
# -----------------------------------
def build_ml_dataset(df):
    df_proc = df.copy()
    df_proc["gejala_list"] = df_proc["gejala"].apply(split_gejala)
    df_proc = df_proc[df_proc["gejala_list"].map(len) > 0]

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df_proc["gejala_list"])

    le = LabelEncoder()
    y = le.fit_transform(df_proc["tanaman"])

    return X, y, mlb, le, df_proc

# -----------------------------------
# NLP EKSTRAKSI GEJALA
# -----------------------------------
def extract_symptoms(text, vocab):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    found = []
    for g in sorted(vocab, key=len, reverse=True):
        if g in text:
            found.append(g)
            text = text.replace(g, " ")

    return found

# -----------------------------------
# MAIN APP
# -----------------------------------
st.title("Manfaat Tanaman Obat Ciplukan & Sirih Cina")
st.markdown(
    "Aplikasi ini menggunakan **Machine Learning (Random Forest)** dan "
    "**NLP sederhana** untuk memprediksi tanaman obat berdasarkan gejala."
)

df = load_dataset(DATA_PATH)

X, y, mlb, le_tanaman, df_proc = build_ml_dataset(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

symptom_vocab = sorted({g for row in df_proc["gejala_list"] for g in row})

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.header("Informasi Model")
st.sidebar.write(f"Jumlah data: {len(df)}")
st.sidebar.write(f"Jumlah gejala unik: {len(symptom_vocab)}")
st.sidebar.write(f"Akurasi model: {acc:.3f}")

menu = st.sidebar.selectbox(
    "Menu",
    ["Dashboard", "Prediksi", "Informasi Data", "Tentang"]
)

# -----------------------------------
# DASHBOARD
# -----------------------------------
if menu == "Dashboard":
    st.header("Ringkasan Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Tanaman")
        fig, ax = plt.subplots()
        df["tanaman"].value_counts().plot.pie(
            autopct="%1.1f%%",
            ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        st.subheader("Frekuensi Gejala Teratas")
        all_gejala = [g for lst in df_proc["gejala_list"] for g in lst]
        top = pd.Series(all_gejala).value_counts().head(15)

        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.barplot(x=top.values, y=top.index, ax=ax2)
        st.pyplot(fig2)

    st.subheader("Contoh Data")
    st.dataframe(df.head(10))

# -----------------------------------
# PREDIKSI
# -----------------------------------
elif menu == "Prediksi":
    st.header("Prediksi Tanaman Berdasarkan Gejala")

    user_text = st.text_area(
        "Masukkan keluhan (contoh: saya mengalami nyeri sendi dan peradangan):"
    )

    manual = st.multiselect(
        "Pilih gejala secara manual:",
        symptom_vocab
    )

    extracted = extract_symptoms(user_text, symptom_vocab)
    combined = list(dict.fromkeys(extracted + manual))

    st.write("Gejala terdeteksi:", combined)

    if combined and st.button("Prediksi"):
        X_input = mlb.transform([combined])
        probs = model.predict_proba(X_input)[0]

        tanaman = le_tanaman.inverse_transform(
            np.arange(len(probs))
        )

        hasil = pd.DataFrame({
            "Tanaman": tanaman,
            "Probabilitas": probs
        }).sort_values("Probabilitas", ascending=False)

        st.table(hasil)
        st.success(f"Rekomendasi utama: **{hasil.iloc[0]['Tanaman']}**")

# -----------------------------------
# INFORMASI DATA
# -----------------------------------
elif menu == "Informasi Data":
    st.header("Pencarian Data Ilmiah")

    keyword = st.text_input("Cari berdasarkan gejala, manfaat, atau referensi:")

    if keyword:
        key = keyword.lower()
        mask = df.apply(
            lambda r:
                key in r["gejala"].lower() or
                key in r["manfaat"].lower() or
                key in r["referensi"].lower(),
            axis=1
        )
        st.dataframe(df[mask])

# -----------------------------------
# ABOUT
# -----------------------------------
else:
    st.header("Tentang Aplikasi")
    st.markdown("""
    - Dataset berbasis **review 26 jurnal ilmiah**
    - Fokus: **Ciplukan (Physalis angulata)** dan **Sirih Cina (Peperomia pellucida)**
    - Metode: **Random Forest + Multi-label Gejala**
    - Tujuan: edukasi dan penelitian
    """)
