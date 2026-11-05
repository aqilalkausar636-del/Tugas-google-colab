"""
Aplikasi Streamlit: Klasifikasi Anomali dan Serangan Jaringan (UNSW-NB15) dengan Random Forest
Proyek 01, Machine Learning for Network Data, D4 TRKJ Universitas Bung Hatta

Jalankan lokal:  streamlit run app.py

PERUBAHAN DARI TEMPLATE:
1. Identitas mahasiswa ditambahkan di bawah judul halaman.
2. Tabel "10 flow paling berisiko" (probabilitas serangan tertinggi).
3. Filter tabel hasil per flow (Semua / SERANGAN / normal) dan tombol unduh
   mengikuti hasil filter.
4. Tab baru "Eksplorasi data": distribusi kategori, perbandingan fitur normal vs serangan,
   dan grafik dampak threshold (precision, recall, akurasi).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
SAMPLE_CSV = BASE / "data" / "sample_flows.csv"

# Isi identitas kamu di sini
NAMA = "Muhammad Aqil Alkausar"
NIM = "ISI_NIM_KAMU"

st.set_page_config(page_title="Deteksi Serangan UNSW-NB15", page_icon="🛡️", layout="wide")


# ---------- Muat model (sekali saja, lalu di-cache) ----------
@st.cache_resource
def load_models():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    rf_bin = joblib.load(MODEL_DIR / "rf_binary.joblib")
    rf_cat = joblib.load(MODEL_DIR / "rf_multiclass.joblib")
    return rf_bin, rf_cat, meta


rf_bin, rf_cat, meta = load_models()
FEATURES = meta["features"]


def predict(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Lengkapi kolom yang hilang dengan nilai default, lalu prediksi dengan kedua model."""
    X = df.copy()
    for col in FEATURES:
        if col not in X.columns:
            X[col] = meta["defaults"][col]
    X = X[FEATURES]
    for col in meta["cat_cols"]:
        X[col] = X[col].astype(str)
    prob = rf_bin.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "prob_serangan": prob.round(3),
        "prediksi": ["SERANGAN" if p >= threshold else "normal" for p in prob],
        "kategori_prediksi": rf_cat.predict(X),
    })
    return out


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    threshold = st.slider("Threshold probabilitas serangan", 0.05, 0.95, 0.50, 0.05,
                          help="Flow dianggap serangan bila probabilitasnya ≥ threshold. "
                               "Turunkan untuk menangkap lebih banyak serangan (recall naik), "
                               "dengan risiko alarm palsu bertambah.")
    st.divider()
    st.subheader("Performa model (data test)")
    mb, mc = meta["metrics_binary"], meta["metrics_multiclass"]
    c1, c2 = st.columns(2)
    c1.metric("Akurasi biner", f"{mb['accuracy']:.1%}")
    c2.metric("ROC-AUC", f"{mb['roc_auc']:.3f}")
    c1.metric("Recall serangan", f"{mb['recall']:.1%}")
    c2.metric("Akurasi kategori", f"{mc['accuracy']:.1%}")
    st.caption(f"Dataset: {meta['dataset']}")

# ---------- Halaman utama ----------
st.title("🛡️ Deteksi Anomali dan Serangan Jaringan")
# PERUBAHAN 1: identitas mahasiswa
st.caption(f"Dibuat oleh: **{NAMA}** | NIM: {NIM} | D4 TRKJ Universitas Bung Hatta")
st.write("Model **Random Forest** yang dilatih pada dataset **UNSW-NB15** memprediksi apakah sebuah "
         "flow jaringan normal atau serangan, sekaligus menebak kategori serangannya.")

# PERUBAHAN 4: tab tambahan "Eksplorasi data"
tab_csv, tab_manual, tab_info, tab_eksplor = st.tabs(
    ["📄 Prediksi dari CSV", "✍️ Input manual", "ℹ️ Tentang model", "📊 Eksplorasi data"])

# --- Tab 1: CSV ---
with tab_csv:
    st.write("Unggah file CSV berisi flow dengan kolom seperti dataset UNSW-NB15, "
             "atau pakai 300 flow contoh dari data test.")
    uploaded = st.file_uploader("File CSV", type="csv")
    use_sample = st.toggle("Gunakan data contoh", value=uploaded is None)

    df = None
    if uploaded is not None and not use_sample:
        df = pd.read_csv(uploaded)
    elif use_sample:
        df = pd.read_csv(SAMPLE_CSV)

    if df is not None:
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.warning(f"{len(missing)} kolom tidak ada dan diisi nilai default (median data latih): "
                       f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
        res = predict(df, threshold)
        shown = pd.concat([res, df.drop(columns=[c for c in res.columns if c in df.columns])], axis=1)

        n_att = int((res["prediksi"] == "SERANGAN").sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Jumlah flow", len(res))
        k2.metric("Diprediksi serangan", n_att)
        k3.metric("Persentase serangan", f"{n_att / len(res):.1%}")

        if "label" in df.columns:
            acc = ((res["prediksi"] == "SERANGAN").astype(int) == df["label"]).mean()
            st.info(f"File memuat kolom `label`, jadi prediksi bisa dicek: akurasi biner = **{acc:.1%}**")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Kategori yang diprediksi")
            kat = res["kategori_prediksi"].value_counts().rename_axis("kategori").reset_index(name="jumlah")
            st.bar_chart(kat, x="kategori", y="jumlah", horizontal=True, sort="-jumlah")
        with right:
            st.subheader("Sebaran probabilitas serangan")
            st.bar_chart(pd.cut(res["prob_serangan"], bins=[0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
                         .astype(str).value_counts().sort_index())

        # PERUBAHAN 2: 10 flow dengan probabilitas serangan tertinggi
        st.subheader("🔥 10 flow paling berisiko")
        st.caption("Flow dengan probabilitas serangan tertinggi, cocok diprioritaskan untuk diselidiki.")
        st.dataframe(shown.sort_values("prob_serangan", ascending=False).head(10),
                     width="stretch")

        # PERUBAHAN 3: filter tabel hasil per flow
        st.subheader("Hasil per flow")
        filter_opt = st.radio("Tampilkan flow", ["Semua", "SERANGAN", "normal"], horizontal=True)
        if filter_opt != "Semua":
            shown = shown[shown["prediksi"] == filter_opt]
        st.caption(f"Menampilkan {len(shown)} flow")
        st.dataframe(shown, width="stretch", height=360)
        st.download_button("⬇️ Unduh hasil (CSV)", shown.to_csv(index=False).encode(),
                           "hasil_prediksi.csv", "text/csv")

# --- Tab 2: Input manual ---
with tab_manual:
    st.write("Pilih flow contoh sebagai titik awal, lalu ubah fitur-fitur terpenting dan tekan **Prediksi**. "
             "Fitur yang tidak ditampilkan memakai nilai dari flow contoh tersebut.")
    samples = pd.read_csv(SAMPLE_CSV)
    pilihan = st.selectbox("Mulai dari contoh flow kategori", sorted(samples["attack_cat"].unique()),
                           index=sorted(samples["attack_cat"].unique()).index("Normal"))
    base = samples[samples["attack_cat"] == pilihan].iloc[0]
    with st.form("manual"):
        cols = st.columns(3)
        row = {c: base[c] for c in FEATURES}
        for i, c in enumerate(meta["cat_cols"]):
            opts = sorted(set(meta["cat_options"][c]) | {str(base[c])})
            row[c] = cols[i].selectbox(c, opts, index=opts.index(str(base[c])), key=f"{pilihan}-{c}")
        cols = st.columns(4)
        for i, c in enumerate(meta["top_numeric"]):
            lo, hi = meta["num_ranges"][c]
            row[c] = cols[i % 4].number_input(c, min_value=float(lo), value=float(base[c]), key=f"{pilihan}-{c}",
                                              help=f"Rentang umum data latih: {lo:g} sampai {hi:g}")
        submitted = st.form_submit_button("🔍 Prediksi", type="primary")
    if submitted:
        r = predict(pd.DataFrame([row]), threshold).iloc[0]
        if r["prediksi"] == "SERANGAN":
            st.error(f"**SERANGAN** terdeteksi (probabilitas {r['prob_serangan']:.1%}). "
                     f"Kategori paling mungkin: **{r['kategori_prediksi']}**")
        else:
            st.success(f"**Normal** (probabilitas serangan {r['prob_serangan']:.1%})")

# --- Tab 3: Info model ---
with tab_info:
    st.subheader("Fitur terpenting (model biner)")
    imp = pd.Series(meta["importance_top15"]).rename_axis("fitur").reset_index(name="importance")
    st.bar_chart(imp, x="fitur", y="importance", horizontal=True, sort="-importance")
    st.subheader("Metrik pada data test")
    st.table(pd.DataFrame({
        "Model biner": {k: round(v, 4) for k, v in meta["metrics_binary"].items()},
    }))
    st.table(pd.DataFrame({
        "Model multikelas": {k: round(v, 4) for k, v in meta["metrics_multiclass"].items()},
    }))
    st.caption(f"Dilatih dengan Python {meta['versions']['python']}, "
               f"scikit-learn {meta['versions']['scikit-learn']}, numpy {meta['versions']['numpy']}.")

# --- Tab 4: Eksplorasi data (tab tambahan) ---
with tab_eksplor:
    st.write("Jelajahi 300 flow contoh dari data test: sebaran kategori serangan, perbandingan fitur "
             "antara trafik normal dan serangan, serta dampak threshold terhadap performa deteksi.")
    data = pd.read_csv(SAMPLE_CSV)
    data["kelompok"] = data["attack_cat"].apply(lambda x: "Normal" if x == "Normal" else "Serangan")

    e1, e2, e3 = st.columns(3)
    e1.metric("Jumlah flow contoh", len(data))
    e2.metric("Flow normal", int((data["kelompok"] == "Normal").sum()))
    e3.metric("Flow serangan", int((data["kelompok"] == "Serangan").sum()))

    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader("Distribusi kategori")
        dist = data["attack_cat"].value_counts().rename_axis("kategori").reset_index(name="jumlah")
        st.bar_chart(dist, x="kategori", y="jumlah", horizontal=True, sort="-jumlah")
    with kanan:
        st.subheader("Fitur kategorikal: normal vs serangan")
        cat_pilih = st.selectbox("Pilih fitur", meta["cat_cols"], key="eksplor-cat")
        st.bar_chart(pd.crosstab(data[cat_pilih].astype(str), data["kelompok"]))

    st.subheader("Rata-rata fitur numerik: normal vs serangan")
    fitur_pilih = st.multiselect("Pilih fitur numerik", meta["top_numeric"],
                                 default=meta["top_numeric"][:5], key="eksplor-num")
    if fitur_pilih:
        st.dataframe(data.groupby("kelompok")[fitur_pilih].mean().T.round(3), width="stretch")

    st.subheader("Dampak threshold pada data contoh")
    st.caption("Semakin rendah threshold, recall naik tetapi alarm palsu (precision turun) bertambah.")
    prob = predict(data, 0.5)["prob_serangan"]
    benar = (data["kelompok"] == "Serangan")
    baris = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = prob >= t
        tp = int((pred & benar).sum())
        fp = int((pred & ~benar).sum())
        fn = int((~pred & benar).sum())
        baris.append({
            "threshold": t,
            "precision": round(tp / (tp + fp), 3) if (tp + fp) else 0.0,
            "recall": round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
            "akurasi": round(float((pred == benar).mean()), 3),
        })
    sweep = pd.DataFrame(baris)
    st.line_chart(sweep, x="threshold", y=["precision", "recall", "akurasi"])
    st.dataframe(sweep, width="stretch", hide_index=True)